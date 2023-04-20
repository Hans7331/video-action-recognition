import torch
from torch import nn, einsum
from torch.nn import functional as F
import torchvision as tv
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
import torchvision.models as models
from collections import OrderedDict





class TubeletEmbeddings(nn.Module):
    """
    Video to Tubelet Embedding.
    """

    def __init__(self, video_size, patch_size, num_channels=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.num_patches = (
            (video_size[2] // patch_size[2])
            * (video_size[1] // patch_size[1])
            * (video_size[0] // patch_size[0])
        )
        self.embed_dim = embed_dim

        self.projection = nn.Conv3d(
            num_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        #self.projection.weight.data = inflate_2d_filter_to_3d(patch_weight, video_size[0], num_channels, embed_dim, patch_size[1], patch_size[2])

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, num_frames, height, width = pixel_values.shape
        x = self.projection(pixel_values) #.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    
    # dim - inner dims of embedding , inner_dim - dim of the transformer
    def __init__(self, dim, inner_dim):
        super().__init__()
        # mlp with GELU activation function
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            #nn.Linear(inner_dim, inner_dim),
            #nn.GELU(),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):
    """
        dim: (int) - inner dimension of embeddings[default:192] 
        heads: (int) - number of attention heads[default:12] # for pretrained model
        dim_head: (int) - dimension of transformer head [default:64] 
    
    """

    def __init__(self, dim = 768, heads = 12, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads 
        #project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # nn.Linear from 192 to (8*64)*3
        self.make_qkv = nn.Linear(dim, inner_dim *3) 

        # Linear projection to required output dimension
        self.get_output = nn.Sequential(nn.Linear(inner_dim, dim))
        #if project_out else nn.Identity()
        

    def forward(self, x):

        b, n, _ = x.shape   # b=batch_size , n=197  ,where n is the input after converting the raw input and adding cls token
        h = self.heads      # h=8

        # nn.Linear from 192 to 256*3 & then split it across q,k & v each with last dimension as 256
        qkv = self.make_qkv(x).chunk(3, dim = -1)
        
        # reshaping to get the right q,k,v dimensions having 8 attn_heads(h)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # dot product of q & k after transposing k followed by a softmax layer
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale     # q.kT / sqrt(d)
        attn = dots.softmax(dim=-1)

        # dot product of attention layer with v
        output = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Final reshaping & nn.Linear to combine all attention head outputs to get final out.
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        output =  self.get_output(output)    
        # output shape = ( b, n, dim (=192) )

        return output
    
class Transformer(nn.Module):
    """ 
        dim: (int) - inner dimension of embeddings 
        depth: (int) - depth of the transformer 
        heads: (int) - number of attention heads [default:16] 
        dim_head: (int) - dimension of transformer head [default:64] 
        mlp_dim: (int) - scaling dimension for attention [default:768] 
    
    """

    def __init__(self, dim, depth, heads=8, dim_head=64, mlp_dim=3072):
        super().__init__()
        
        self.model_layers = nn.ModuleList([])
        for i in range(depth):
            self.model_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                #nn.MultiheadAttention(dim, heads,batch_first=True),
                Attention(dim, heads, dim_head),
                nn.LayerNorm(dim),
                MLP(dim, mlp_dim)
            ]))

        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x):

        for layer_norm1, attention, layer_norm2, ff_net in self.model_layers:
            
            x = attention(layer_norm1(x)) + x
            x = ff_net(layer_norm2(x)) + x


        return self.layer_norm(x)



class ViViT_2(nn.Module):
    """  
    Args:
        image_size: (int) - size of input image
        patch_size: (int) - size of each patch of the image 
        num_classes: (int) - number of classes in the dataset
        frames_per_clip: (int) - number of frames in every video clip. [default:16] 
        dim: (int) - inner dimension of embeddings[default:192] 
        depth: (int) - depth of the transformer[default:4] 
        heads: (int) - number of attention heads for the transformer[default:12] 
        pooling: (str) - type of pooling[default:'mean'] 
        in_channels: (int) - number of input channels for each frame [default:3] 
        dim_head: (int) - dimension of transformer head [default:64] 
        scale_dim: (int) - scaling dimension for attention [default:4] 
    
    """

    def __init__(self, image_size, patch_size, num_classes, frames_per_clip=32, dim = 768, depth = 4, heads = 12, pooling = 'mean', in_channels = 3, dim_head = 64, scale_dim = 4 ):
        
        super().__init__()

        num_patches = (image_size // patch_size) ** 2   # => 196 for 224x224 images
        patch_dim = in_channels * patch_size ** 2      # => 3*16*16


        self.tube = TubeletEmbeddings((32,3, 224,224), (1,16,16), num_channels=3, embed_dim=768)

        # position embeddings of shape: (1, frames_per_clip = 16, num_patches + 1 = 197, 192)
        self.pos_embed = nn.Parameter(torch.randn(1, frames_per_clip, num_patches + 1, dim))

        # space (i.e. for each image) tokens of shape: (1, 1, 192). The 192 is the tokens obtained in "get_patch_emb" 
        self.spatial_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # spatial transformer ViT
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim)

        # time dimention tokens of shape: (1, 1, 192). 
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # temporal transformer which takes in spacetransformer's output tokens as the input. 
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim)

        # pooling type, could be "mean" or "cls"
        self.pooling = pooling

        # mlp head for final classification
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            #nn.Softmax(dim=1)
        )

        self.downsample = nn.Sequential(
            nn.Conv3d(3, 64, (1, 2, 2), (1, 2, 2) , bias = False),
            nn.Conv3d(64, 128, (1, 2, 2), (1, 2, 2) , bias = False),
            nn.Conv3d(128,256, (1, 2, 2), (1, 2, 2) , bias = False),
            nn.Conv3d(256, 512, (1, 2, 2), (1, 2, 2) , bias = False),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        self.fc1 = nn.Linear(512,512, bias = True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 128, bias = False)

        self.temp_avg = nn.AdaptiveAvgPool3d((1,None,None))

    def forward(self, x):

        
        x, clip_type = x
        x_init = x.clone()
        if clip_type == 'd':
        
            # get patch embeddings
            #print("orignal shape", x.shape)
            x = x.permute(0,2,1,3,4)
            #print("shape after permute: ", x.shape)
            x = self.tube(x)
            #print("shape after tube: ", x.shape)
            x =  rearrange(x, 'b c t h w -> b t (h w) c')
            #print("shape after rearrange: ", x.shape)
            #x = self.get_patch_emb(x)

            # b = batch_size , t = frames , n = number of patch embeddings= 14*14 , e = embedding size
            b, t, n, e = x.shape     # x.shape = (b, t, 196, 192) 

            # prepare cls_token for space transformers
            spatial_cls_tokens = repeat(self.spatial_token, '() n d -> b t n d', b = b, t=t)

            # concatenate cls_token to the patch embedding
            x = torch.cat((spatial_cls_tokens, x), dim=2)     # => x shape = ( b, t, 197 ,192)

            # add position embedding info 
            x += self.pos_embed[:, :, :(n + 1)]

            # club together the b & t dimension 
            x = rearrange(x, 'b t n d -> (b t) n d')

            # pass through spatial transformer
            x = self.spatial_transformer(x)

            # declub b & t dimensions
            x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

            # prepare cls_token for temporal transformers & concatenate cls_token to the patch embedding
            temporal_cls_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
            x = torch.cat((temporal_cls_tokens, x), dim=1)
            # pass through spatial transformer
            x = self.temporal_transformer(x)
            # if pooling is mean, then use mean of all 197 as the output, else use output corresponding to cls token as the final x
            if self.pooling == 'mean':
                x = x.mean(dim = 1) #( b, n, dim (=192) )
            else:
                x[:, 0] #( b, n, dim (=192) )
            # pass through MLP classification layer
            x = self.classifier_head(x)
        
        ##### sparse augmentation
        if clip_type == 's':
            
            #x_init = self.downsample(x_init.permute(0,2,1,3,4))
            x = self.temp_avg(x)
            x = x.flatten(1)
            
            x = self.relu(self.bn1(self.fc1(x)))
            x = nn.functional.normalize(self.bn2(self.fc2(x)), p=2, dim=1)

            x1, x2, x3, x4 = [nn.functional.normalize(self.bn2(self.fc2(\
                                    self.relu(self.bn1(self.fc1(x_init[:,:,i,:,:].flatten(1))))))) for i in range(4)]
            
            return x, x1, x2, x3, x4

        return x
