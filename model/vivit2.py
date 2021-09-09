import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.vivit_module import Attention, PreNorm, FeedForward
import numpy as np

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


  
class ViViT(nn.Module):

    def __init__(self, image_size=256, patch_size=16, tubelet_temporal_size=2, num_classes=2, num_frames=32, dim=192, layer_spacial=12, layer_temporal=4, heads=3, pool='cls', in_channels=3, dim_head=64, dropout=0.2, emb_dropout=0.1, mlp_dim=2048, pretrain=False):
        # (ViT-B,
        #   L(depth, the number of trans- former layers)=12,
        #   NH(heads, each with a self-attention block of NH heads)=12,
        #   d(hidden dimension d)=3072)
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_size = patch_size
        self.tubelet_temporal_size = tubelet_temporal_size
        self.in_channels = in_channels

        num_patches = (image_size // patch_size) ** 2
        # patch_dim = in_channels * patch_size ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
        #     nn.Linear(patch_dim, dim),
        # )

        tubelet_dim = self.tubelet_temporal_size * self.patch_size * self.patch_size * self.in_channels
        self.to_tubelet_embedding = nn.Sequential(
            Rearrange('b (t pt) c (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=self.tubelet_temporal_size, ph=self.patch_size, pw=self.patch_size),
            nn.Linear(tubelet_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames//self.tubelet_temporal_size, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, layer_spacial, heads, dim_head, mlp_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, layer_temporal, heads, dim_head, mlp_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        if pretrain:
            state_dict = torch.load(pretrain)
            self.pos_embedding = nn.Parameter(repeat(state_dict['pos_embed'], '() n d -> b t n d', b=1, t=num_frames//self.tubelet_temporal_size))
            # init spacial transformer with vit-b weights
            for i in range(layer_spacial):
                self.space_transformer.layers[i][0].norm.weight = nn.Parameter(
                    state_dict['blocks.{}.norm1.weight'.format(i)])
                self.space_transformer.layers[i][0].norm.bias = nn.Parameter(
                    state_dict['blocks.{}.norm1.bias'.format(i)])

                self.space_transformer.layers[i][0].fn.to_qkv.weight = nn.Parameter(
                    state_dict['blocks.{}.attn.qkv.weight'.format(i)])
                self.space_transformer.layers[i][0].fn.to_qkv.bias = nn.Parameter(
                    state_dict['blocks.{}.attn.qkv.bias'.format(i)])

                self.space_transformer.layers[i][0].fn.to_out[0].weight = nn.Parameter(
                    state_dict['blocks.{}.attn.proj.weight'.format(i)])
                self.space_transformer.layers[i][0].fn.to_out[0].bias = nn.Parameter(
                    state_dict['blocks.{}.attn.proj.bias'.format(i)])



                self.space_transformer.layers[i][1].norm.weight = nn.Parameter(
                    state_dict['blocks.{}.norm2.weight'.format(i)])
                self.space_transformer.layers[i][1].norm.bias = nn.Parameter(
                    state_dict['blocks.{}.norm2.bias'.format(i)])

                self.space_transformer.layers[i][1].fn.net[0].weight = nn.Parameter(
                    state_dict['blocks.{}.mlp.fc1.weight'.format(i)])
                self.space_transformer.layers[i][1].fn.net[0].bias = nn.Parameter(
                    state_dict['blocks.{}.mlp.fc1.bias'.format(i)])

                self.space_transformer.layers[i][1].fn.net[3].weight = nn.Parameter(
                    state_dict['blocks.{}.mlp.fc2.weight'.format(i)])
                self.space_transformer.layers[i][1].fn.net[3].bias = nn.Parameter(
                    state_dict['blocks.{}.mlp.fc2.bias'.format(i)])


            # init temporal transformer to zero
            for param in self.temporal_transformer.parameters():
                torch.nn.init.zeros_(param)
                
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, 
        #             gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # x = self.to_patch_embedding(x)
        x = self.to_tubelet_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0, :], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0, :]

        return self.mlp_head(x)
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 16, 3, 224, 224]).cuda()
    
    model = ViViT(224, 16, 100, 16).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]