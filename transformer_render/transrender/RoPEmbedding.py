import torch
import torch.nn as nn
from einops import rearrange
from transrender.RMSNorm import RMSNorm
from .SwiGLU import SwiGLU
import torch.nn.functional as F
import math

def get_triangle_rope_embeddings(vertices, d_head, n_freqs=6):
    """
    根据论文3.1节 "Relative Spatial Positional Embedding" 实现。
    此版本经过健壮性重构，以确保输出维度始终为 d_head。

    Args:
        vertices (torch.Tensor): 三角面片顶点坐标 [B, N_tri, 3, 3]
        d_head (int): 每个头的维度，必须是偶数
        n_freqs (int): 用于顶点位置的频率数量
    Returns:
        torch.Tensor: RoPE嵌入 [B, N_tri, d_head]
    """
    if d_head % 2 != 0:
        raise ValueError(f"RoPE embedding dimension (d_head) must be even, but got {d_head}")

    # 1. 展平顶点 -> 9D 向量
    flat_vertices = rearrange(vertices, 'b n v d -> b n (v d)') # [B, N_tri, 9]
    
    # 2. 创建频率
    # 论文中频率从1到5，这里用 logspace 生成
    freq_bands = torch.logspace(0.0, math.log10(5.0), n_freqs, device=vertices.device)
    
    # 3. 将9D顶点乘以6个频率，得到54个scaled frequencies
    # flat_vertices: [B, N, 9, 1], freq_bands: [6]
    scaled_freqs = flat_vertices.unsqueeze(-1) * freq_bands
    scaled_freqs = rearrange(scaled_freqs, 'b n d f -> b n (d f)') # [B, N, 54]
    
    # 4. 根据d_head决定使用多少个频率
    num_freq_pairs_needed = d_head // 2
    
    if num_freq_pairs_needed > scaled_freqs.shape[-1]:
        raise ValueError(
            f"d_head ({d_head}) is too large. "
            f"With 9D vertices and {n_freqs} freqs, we can generate a max of {scaled_freqs.shape[-1]*2} RoPE dimensions."
        )
        
    # 5. 从54个频率中取出我们需要的部分
    freqs_for_rope = scaled_freqs[:, :, :num_freq_pairs_needed] # [B, N, d_head/2]
    
    # 6. 计算sin和cos
    sin_embeds = torch.sin(freqs_for_rope)
    cos_embeds = torch.cos(freqs_for_rope)
    
    # 7. 交错拼接sin和cos，得到最终的RoPE嵌入
    # stack -> [B, N, d_head/2, 2]
    # rearrange -> [B, N, d_head]
    rope_embeds = torch.stack([sin_embeds, cos_embeds], dim=-1)
    rope_embeds = rearrange(rope_embeds, 'b n d c -> b n (d c)')
    
    return rope_embeds

def apply_rope(x, rope_embeds):
    
    if not isinstance(rope_embeds, torch.Tensor):
        raise TypeError(
            f"[apply_rope] FATAL: Expected rope_embeds to be a torch.Tensor, but got {type(rope_embeds)}."
            "This indicates a critical logic error in how RoPE embeddings are passed."
        )
    """
    应用RoPE。
    
    Args:
        x (torch.Tensor): Q或K张量, 形状为 [B, H, N, D_head]
        rope_embeds (torch.Tensor): RoPE嵌入, 形状为 [B, N, D_head]
    Returns:
        torch.Tensor: 应用RoPE后的张量, 形状为 [B, H, N, D_head]
    """
    # 将 rope_embeds reshape 成复数形式
    # rope_embeds: [B, N, D_head] -> [B, N, D_head/2] (复数)
    rope_complex = torch.view_as_complex(
        rearrange(rope_embeds, 'b n (d c) -> b n d c', c=2).contiguous()
    )
    
    # 将 x reshape 成复数形式
    # x: [B, H, N, D_head] -> [B, H, N, D_head/2] (复数)
    x_complex = torch.view_as_complex(
        rearrange(x, 'b h n (d c) -> b h n d c', c=2).contiguous()
    )
    
    # 广播 rope_complex 到所有头
    # rope_complex: [B, N, D_head/2] -> [B, 1, N, D_head/2]
    # x_complex:    [B, H, N, D_head/2]
    # 乘法将在 H 维度上进行广播
    rope_complex_broadcasted = rope_complex.unsqueeze(1)
    
    x_rotated = x_complex * rope_complex_broadcasted
    
    # 转换回实数张量
    x_out = torch.view_as_real(x_rotated)
    x_out = rearrange(x_out, 'b h n d c -> b h n (d c)')
    
    return x_out.contiguous()


class AttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model)
        
    def forward(self, x, context=None,q_rope=None, k_rope=None):
        # x: [B, N_query, D]
        # context: [B, N_kv, D] (可选, 用于交叉注意力)
        # rope_embeds: [B, N, D_head]
        
        kv_input = context if context is not None else x
        
        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)
        
        # QK Normalization (论文提及)
        q = q * (self.d_head ** -0.5)
        
        # 应用RoPE
        if q_rope is not None:
            q = apply_rope(q, q_rope)
        if k_rope is not None:
            k = apply_rope(k, k_rope)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class RenderFormerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = AttentionWithRoPE(d_model, n_heads)
        
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = AttentionWithRoPE(d_model, n_heads)
        
        self.norm3 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim * 2),
            SwiGLU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x, context=None, x_rope=None, context_rope=None):
        # 视图相关阶段 (Decoder-like)
        if context is not None:
            # Cross-Attention
            x = x + self.cross_attn(self.norm2(x), context, q_rope=x_rope, k_rope=context_rope)
            # Self-Attention
            x = x + self.attn(self.norm1(x), q_rope=x_rope, k_rope=x_rope)
        # 视图无关阶段 (Encoder-like)
        else:
            x = x + self.attn(self.norm1(x), q_rope=x_rope, k_rope=x_rope)
            
        # Feed-Forward
        x = x + self.ffn(self.norm3(x))
        return x