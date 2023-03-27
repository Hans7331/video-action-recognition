import torch
from torch import nn, einsum
from torch.nn import functional as F
import torchvision as tv
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from collections import OrderedDict
from pytorch_pretrained_vit import ViT

# position embedding technique(explained in section 3.4)

def pretrained_pos_embedding(frames_per_clip):
    checkpoint = torch.load('vit.pth',map_location=torch.device('cpu'))
    pos_embed_weights = OrderedDict()
    for key,value in checkpoint.items():
        if key.startswith('pos_embed'):
            pos_embed_weights[key] = value
    print(pos_embed_weights.keys())
    x  = pos_embed_weights.values()
    y = next(iter(x))
    y =  repeat(y, 'v n d -> v f n d', f = frames_per_clip)
    pos_embed_weights['pos_embed'] = y
    #torch.save(pos_embed_weights, 'pos_embed.pt')
    return y


#implementing the tubelet embedding and patch initialising from the Vit weights

model= ViT('B_16_imagenet1k', pretrained=True)
patch_weight = model.patch_embedding.weight

def inflate_2d_filter_to_3d(filter_2d, num_frames, num_channels_in, num_channels_out, filter_height, filter_width):
    # Initialize a 3D filter with zeros
    filter_3d = torch.zeros((num_channels_out ,num_channels_in,num_frames, filter_height, filter_width))


    # Inflate the 2D filter by replicating it along the temporal dimension and averaging them
    filter_2d_1 = filter_2d.clone().detach().cpu().numpy()
    
    for i in range(num_frames):
        filter_3d[:, :, i, :, :] = torch.tensor(filter_2d_1).float()/ num_frames

    return filter_3d


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
        self.projection.weight.data = inflate_2d_filter_to_3d(patch_weight, video_size[0], num_channels, embed_dim, patch_size[1], patch_size[2])

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, num_frames, height, width = pixel_values.shape
        x = self.projection(pixel_values) #.flatten(2).transpose(1, 2)
        return x


# FE Vivit Model 

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

    
## Main ViViT model (2)
class ViViT_2(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, frames_per_clip , dim = 768, depth = 4, heads = 3, 
        pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., scale_dim = 4,tube = False):
        super().__init__()
        
        # mlp_dim = 3072
        # num_layers = 12 ( spatial) 4 ( temporal) [is depth]
        # num_heads = 12


        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        self.tube = tube
        self.tubelet_emb = TubeletEmbeddings((16,3, 224, 224), (16,16,16), num_channels=3, embed_dim=768)
        
        # here 1 is tubelet size instead of number of frames-per clip,change to frames_per_clip when using patch embedding
        self.pos_embedding = nn.Parameter(pretrained_pos_embedding(1))  
        #self.pos_embedding = nn.Parameter(torch.randn(1, frames_per_clip, num_patches + 1, dim)) # <-
        
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        if self.tube==True:
            x = x.permute(0,2,1,3,4)
            x = self.tubelet_emb(x)
            x =  rearrange(x, 'b c t h w -> b t (h w) c')
        else:
            x = self.to_patch_embedding(x)
        
        b, t, n, _ = x.shape
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


class ViViT_1(nn.Module):
    """   
    Args:
        image_size: (int) - size of input image
        patch_size: (int) - size of each patch of the image 
        num_classes: (int) - number of classes in the dataset
        frames_per_clip: (int) - number of frames in every video clip. [default:16] 
        dim: (int) - inner dimension of embeddings[default:192] 
        depth: (int) - depth of the transformer[default:4] 
        heads: (int) - number of attention heads for the transformer[default:8] 
        pooling: (str) - type of pooling[default:'mean'] 
        in_channels: (int) - number of input channels for each frame [default:3] 
        dim_head: (int) - dimension of transformer head [default:64] 
        scale_dim: (int) - scaling dimension for attention [default:4] 
    
    """

    def __init__(self, image_size, patch_size, num_classes, frames_per_clip=16, dim = 192, depth = 4, heads = 8, pooling = 'mean', in_channels = 3, dim_head = 64, scale_dim = 4, ):
        
        super().__init__()

        num_patches = (image_size // patch_size) ** 2   # => 196 for 224x224 images
        patch_dim = in_channels * patch_size ** 2      # => 3*16*16

        self.get_patch_emb = nn.Sequential(
            # input h = 14, w=14, c=3, p1=16, p2=16
            # reshape from (batch_size, frames, channels, 224, 224) to  (batch_size, frames, 14*14, 16*16*3 )
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  

            # fully connected from 16*16*3 to 192 for every patch in (batch_size, frames) 
            nn.Linear(patch_dim, dim),
        )

        # position embeddings of shape: (1, frames_per_clip = 16, num_patches + 1 = 197, 192)
        self.pos_embedding = nn.Parameter(torch.randn(1, frames_per_clip, num_patches + 1, dim))

        # space (i.e. for each image) tokens of shape: (1, 1, 192). The 192 is the tokens obtained in "get_patch_emb" 
        self.spatio_temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # spatial transformer ViT
        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim)

        # pooling type, could be "mean" or "cls"
        self.pooling = pooling

        # mlp head for final classification
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        # get patch embeddings
        x = self.get_patch_emb(x)

        # b = batch_size , t = frames , n = number of patch embeddings= 14*14 , e = embedding size
        b, t, n, e = x.shape     # x.shape = (b, t, 196, 192) 

        # prepare cls_token for space transformers
        spatio_temporal_cls_tokens = repeat(self.spatio_temporal_token, '() n d -> b t n d', b = b, t=t)

        # concatenate cls_token to the patch embedding
        x = torch.cat((spatio_temporal_cls_tokens, x), dim=2)     # => x shape = ( b, t, 197 ,192)

        # add position embedding info 
        x += self.pos_embedding[:, :, :(n + 1)]

        # club together the b & t dimension 
        x = rearrange(x, 'b t n d -> (b t) n d')

        # pass through spatial transformer
        x = self.transformer(x)

        # declub b & t dimensions
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        
        # if pooling is mean, then use mean of all 197 as the output, else use output corresponding to cls token as the final x
        if self.pooling == 'mean':
            x = x.mean(dim = 1) #( b, n, dim (=192) )
        else:
             x[:, 0] #( b, n, dim (=192) )

        # pass through MLP classification layer
        return self.classifier_head(x)
