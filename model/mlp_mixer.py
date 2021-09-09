from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )



# def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
#     assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
#     num_patches = (image_size // patch_size) ** 2
#     chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

#     return nn.Sequential(
#         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
#         nn.Linear((patch_size ** 2) * channels, dim),
#         *[nn.Sequential(
#             PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
#             PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
#         ) for _ in range(depth)],
#         nn.LayerNorm(dim),
#         Reduce('b n c -> b c', 'mean'),
#         nn.Linear(dim, num_classes)
#     )





class MLPMixer(nn.Module):
    def __init__(self, *, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
        super().__init__()

        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        self.mlp_mixer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # m.weight.data = nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, img):
        return self.mlp_mixer(img)