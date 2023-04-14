import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models
​
from einops import rearrange, repeat, reduce
​
# B (b) batch dimension
# R (r) ratio dimension
# C (c) color dimension
# H (h) height dimension
# W (w) width dimension
​
def create_feature_extractor(net_type, name_layer):
    net_list = []
    num_cascade = 0
    num_channel_feature = 0
​
    if net_type == 'vgg16':
        base_model = models.vgg16().features
        for name, layer in base_model.named_children():
            net_list.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                num_cascade += 1
            if isinstance(layer, nn.Conv2d):
                num_channel_feature = layer.out_channels
            if name == name_layer:
                break
        net = nn.Sequential(*net_list)
    elif net_type == 'ef4':
        base_model =  models.efficientnet_b4(weights='IMAGENET1K_V1').features
        for name, layer in base_model.named_children():
            net_list.append(layer)
            if name == name_layer:
                break
        net = nn.Sequential(*net_list)
        num_channel_feature = 160
        num_cascade = 4
    return net, num_channel_feature, num_cascade
​

class BiForwardHead(nn.Module):
    def __init__(self, num_embedding_dim, num_channel_out, num_channel_feature):
        super().__init__()
        self.num_channel_feature = num_channel_feature
​
        self.ARS_FTM_head = nn.Linear(
            num_embedding_dim, num_channel_feature**2)
        self.ARS_PWP_head = nn.Linear(num_embedding_dim, num_channel_out)
​
    def forward(self, x):
        ARS_FTM = rearrange(self.ARS_FTM_head(
            x), 'b r (c1 c2) -> b r c1 c2', c1=self.num_channel_feature)
        ARS_PWP = rearrange(self.ARS_PWP_head(x), 'b r cout -> b r cout () ()')
        return ARS_FTM, ARS_PWP
​
​
class MetaLearner(nn.Module):
    def __init__(self, num_embedding_dim, num_layers, num_channel_out, num_channel_feature, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Linear(num_embedding_dim, num_embedding_dim), nn.ReLU(), nn.Dropout(dropout_rate)]*num_layers, BiForwardHead(
            num_embedding_dim, num_channel_out, num_channel_feature))
​
    def forward(self, x):
        return self.net(x)
​
​
class DeconvBlock(nn.Module):
    def __init__(self, num_channel_in, num_channel_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channel_in, num_channel_out, (3, 3), padding=(1, 1)), nn.BatchNorm2d(num_channel_out), nn.ReLU())
​
    def forward(self, x):
        x = repeat(x, 'br c h w -> br c (h f1) (w f2)', f1=2, f2=2)
        return self.net(x)
​
​
class Deconv(nn.Module):
    def __init__(self, num_cascade, num_channel_out, num_channel_feature):
        super().__init__()
        self.net = nn.Sequential(DeconvBlock(num_channel_feature, num_channel_out),
                                 * [DeconvBlock(num_channel_out, num_channel_out)]*(num_cascade-1))
​
    def forward(self, x):
        return self.net(x)
​
​
class Mars(nn.Module):
    def __init__(self, net_type, name_layer, dropout_rate=0.2, dim_embedding=512, num_embedding=101, num_channel_out=96, num_meta_learner_hidden_layers=2):
        super().__init__()
        """
        Arguments:
        net_type - defines the type of base model used for feature extraction[default:vgg]
        name_layer - number of layers in base model[default:15]
        dropout_rate - prob of training a node in a hidden layer[default:0.2]
        dim_embedding  -  dimension of embedding[default:512]
        num_embedding - number of embedding, linked to number of chosen aspect ratio set[default:101]
        num_channel_out - number of channel in output[default:96]
        num_meta_learner_hidden_layers - ARS-FTM, ARS-PWP[default:2]
        """
        
​
        #self.ratio_embedding_nodes = nn.Parameter(
            #torch.rand(num_embedding, dim_embedding))
        #self.embedding_interp_step = (2*math.log(4))/(num_embedding-1)
        #self.meta_learner = MetaLearner(
            #dim_embedding, num_meta_learner_hidden_layers, num_channel_out, num_channel_feature, dropout_rate)
​
        self.feature_extractor, num_channel_feature, num_cascade = create_feature_extractor(
            net_type, name_layer)
       
        self.deconv_layers = Deconv(
            num_cascade, num_channel_out, num_channel_feature)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_layer = nn.Conv2d(num_channel_out,1,kernel_size=(1,1))
​
    def forward(self, x,ratio):
        #generate ARS_FTM and ARS_PWP given ratio
        #ratio_embedding = self.get_ratio_embedding_batch(ratio)
        #ARS_FTM, ARS_PWP = self.meta_learner(ratio_embedding)
​
        # get the features from image
        x = self.feature_extractor(x)
        #print("feat ext", x.shape)
        h = x.shape[2]
        w = x.shape[3]
        x_gap = self.GAP(x)
        #print("GAP", x_gap.shape)
​
        # repeat to fill the ratio dimension
        #r = ratio.shape[1]
        #x_gap = repeat(x_gap, 'b c () () -> b r c () ()', r=r)
        #print("repeat operation", x_gap.shape)
        # transform and add
        #x_gap = x_gap + self.ratio_transform(x_gap, ARS_FTM)  # b*r*c*1*1 #fars
        #print("ARS_FTM INTEG",x_gap.shape)
​
        
        # replicate to h*w*c
        x_gap = repeat(x_gap, 'b c () () -> b c h w', h=h, w=w)
​
        x = x + x_gap
        #print("replicate h*w*c",x.shape)  #input to deconvolution
        # deconv
        #x = rearrange(x, 'b c h w -> (b) c h w')
        x = self.deconv_layers(x)
        #x = rearrange(x, '(b) c h w -> b c h w')
        #print("Shape after deconvolution",x.shape)
​
        # predict point-wise
        x = self.conv_layer(x)
        #x = self.pixelwise_predict(x, ARS_PWP)
        #print("shape after last 1*1 convolution",x.shape)
        return F.sigmoid(x) 
​
# after GAP fara is generated replicate h*w times the fara and do element-wise addition to obtain input to the deconvolution
​
    def get_ratio_embedding(self, ratio):
        log_ratio = math.log(ratio)
        idx_low_node = math.floor(
            (log_ratio+math.log(4))/self.embedding_interp_step) - 1
        rate_high = (log_ratio - (idx_low_node+1) *
                     self.embedding_interp_step + math.log(4))/self.embedding_interp_step
        ratio_embedding = self.ratio_embedding_nodes[idx_low_node, :]*(
            1-rate_high)+self.ratio_embedding_nodes[idx_low_node+1, :]*rate_high
        ratio_embedding = rearrange(ratio_embedding, 'n -> () () n')
        return ratio_embedding
​
    def get_ratio_embedding_batch(self, batch_ratios):
        ratio_embedding = torch.cat([torch.cat([self.get_ratio_embedding(ratio)
                                                for ratio in ratios], dim=1) for ratios in batch_ratios], dim=0)
        return ratio_embedding
​
    @staticmethod
    def ratio_transform(x, ARS_FTM):
        x = rearrange(x, 'b r c () () -> b r () c')
        x = torch.matmul(x, ARS_FTM)
        x = rearrange(x, 'b r () c -> b r c () ()')
        return x
​
    @staticmethod
    def pixelwise_predict(x, ARS_PWP):
        x = x*ARS_PWP
        x = reduce(x, 'b r c h w -> b r h w', 'sum')
        return x