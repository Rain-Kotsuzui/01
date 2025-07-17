import torch.nn as nn
from einops import rearrange
from .RMSNorm import RMSNorm

class RayBundleEmbedding(nn.Module):
    def __init__(self, d_model=768, patch_size=8):
        super().__init__()
        # 64条光线方向 (3D) -> 192维
        input_dim = patch_size * patch_size * 3 
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            RMSNorm(d_model)
        )

    def forward(self, ray_directions):
        """
        Args:
            ray_directions (torch.Tensor): [B, N_bundles, patch_size, patch_size, 3]
        """
        b, n, _, _, _ = ray_directions.shape
        flat_rays = rearrange(ray_directions, 'b n h w d -> (b n) (h w d)')
        ray_embeds = self.embedding(flat_rays)
        return rearrange(ray_embeds, '(b n) d -> b n d', b=b)