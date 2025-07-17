import torch
import torch.nn as nn
import math
from einops import rearrange
from .RMSNorm import RMSNorm

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_freqs, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_freqs = nn.Parameter(torch.linspace(0., num_freqs - 1, num_freqs) * math.pi, requires_grad=False)
        self.output_dim = input_dim * (1 + 2 * num_freqs) if include_input else input_dim * 2 * num_freqs

    def forward(self, x):
        embeds = [x] if self.include_input else []
        freqs = torch.exp(self.log_freqs)
        for freq in freqs:
            embeds.append(torch.sin(freq * x))
            embeds.append(torch.cos(freq * x))
        return torch.cat(embeds, dim=-1)

class TriangleEmbedding(nn.Module):
    def __init__(self, d_model=768, num_freqs=6):
        super().__init__()
        # 法线嵌入
        normal_encoder = PositionalEncoding(3, num_freqs)
        self.normal_mlp = nn.Sequential(
            normal_encoder,
            nn.Linear(normal_encoder.output_dim * 3, d_model),
            RMSNorm(d_model)
        )
        
        # 材质和发光嵌入 (10维: 3漫反射+3高光+1粗糙度+3发光)
        self.brdf_emission_mlp = nn.Sequential(
            nn.Linear(10, d_model),
            RMSNorm(d_model)
        )

    def forward(self, vertex_normals, brdf_params, emission):
        """
        Args:
            vertex_normals (torch.Tensor): [B, N_tri, 3, 3] (3个顶点的法线)
            brdf_params (torch.Tensor): [B, N_tri, 7] (diffuse, specular, roughness)
            emission (torch.Tensor): [B, N_tri, 3]
        """
        # 1. 处理法线
        b, n, _, _ = vertex_normals.shape
        normals_flat = rearrange(vertex_normals, 'b n v d -> (b n) v d')
        normal_embeds = self.normal_mlp(rearrange(normals_flat, 'bn v d -> bn (v d)'))
        normal_embeds = rearrange(normal_embeds, '(b n) d -> b n d', b=b)
        
        # 2. 处理材质和发光
        brdf_emission = torch.cat([brdf_params, emission], dim=-1) # [B, N, 10]
        brdf_emission_embeds = self.brdf_emission_mlp(brdf_emission)
        
        # 3. 相加得到最终嵌入
        return normal_embeds + brdf_emission_embeds