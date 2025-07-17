import torch
import torch.nn as nn
from einops import rearrange, repeat
from transrender.RMSNorm import RMSNorm
from .TriangleEmbedding import TriangleEmbedding
from .RayEmbedding import RayBundleEmbedding
from .RoPEmbedding import get_triangle_rope_embeddings, RenderFormerBlock
import torch.nn.functional as F

class TransRender(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=6,  # 论文中是6头
                 n_layers_indep=12,
                 n_layers_dep=6,
                 mlp_ratio=4,
                 n_register_tokens=16,
                 patch_size=8):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_register_tokens = n_register_tokens
        self.patch_size = patch_size

        # 1. 输入嵌入模块
        self.triangle_embedding = TriangleEmbedding(d_model)
        self.ray_bundle_embedding = RayBundleEmbedding(d_model, patch_size)
        if self.n_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(self.n_register_tokens, d_model))

        # 2. 视图无关阶段
        self.view_independent_stage = nn.ModuleList([
            RenderFormerBlock(d_model, n_heads, d_model * mlp_ratio)
            for _ in range(n_layers_indep)
        ])

        # 3. 视图相关阶段
        self.view_dependent_stage = nn.ModuleList([
            RenderFormerBlock(d_model, n_heads, d_model * mlp_ratio)
            for _ in range(n_layers_dep)
        ])

        # 4. 输出头 (将光线束Token解码为像素值)
        # 论文说用了一个DPT解码器，这里我们简化为一个MLP，精神是一致的
        output_dim = patch_size * patch_size * 3  # HDR RGB
        self.output_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, triangles,camera):
        """
        Args:
            triangles (dict): 包含三角面片所有信息的字典
                'vertices': [B, N_tri, 3, 3] # 世界坐标
                'vertex_normals': [B, N_tri, 3, 3]
                'brdf_params': [B, N_tri, 7]
                'emission': [B, N_tri, 3]
            camera (dict): 包含相机信息的字典
                'ray_bundles_directions': [B, N_bundles, patch_size, patch_size, 3] # 相机坐标系下的方向
                'world_to_cam_matrix': [B, 4, 4]
        """
        B, N_tri, _, _ = triangles['vertices'].shape
        N_bundles = camera['ray_bundles_directions'].shape[1]

        # --- 视图无关阶段 ---
        # 1.1 生成三角面片嵌入
        tri_tokens = self.triangle_embedding(
            triangles['vertex_normals'],
            triangles['brdf_params'],
            triangles['emission']
        )

        # 1.2 添加寄存器词元
        if self.n_register_tokens > 0:
            reg = repeat(self.register_tokens, 'n d -> b n d', b=B)
            # 论文说寄存器词元使用场景顶点的平均位置来计算RoPE
            avg_pos = triangles['vertices'].mean(
                dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 3]
            reg_vertices = avg_pos.repeat(
                1, self.n_register_tokens, 3, 3)  # [B, 16, 3, 3]

            # 合并词元和它们的虚拟顶点
            context_vertices = torch.cat(
                [reg_vertices, triangles['vertices']], dim=1)
            context_tokens = torch.cat([reg, tri_tokens], dim=1)
        else:
            context_vertices = triangles['vertices']
            context_tokens = tri_tokens

        # 1.3 计算视图无关阶段的RoPE
        context_rope = get_triangle_rope_embeddings(
            context_vertices, self.d_head)

        # 1.4 通过Transformer块
        for block in self.view_independent_stage:
            context_tokens = block(
                context_tokens, x_rope=context_rope)

        # --- 视图相关阶段 ---
        # 2.1 生成光线束嵌入
        ray_tokens = self.ray_bundle_embedding(
            camera['ray_bundles_directions'])

        # 2.2 计算视图相关阶段的RoPE
        # 三角面片顶点需要转换到相机坐标系
        verts_world = F.pad(
            rearrange(context_vertices, 'b n v d -> b (n v) d'), (0, 1), value=1.0)
        verts_cam = verts_world @ camera['world_to_cam_matrix'].transpose(1, 2)
        verts_cam = rearrange(
            verts_cam[:, :, :3], 'b (n v) d -> b n v d', n=context_tokens.shape[1], v=3)

        context_rope_cam = get_triangle_rope_embeddings(verts_cam, self.d_head)

        # 光线束的RoPE。因为在相机空间原点为(0,0,0)，方向为d, 可以认为其顶点为(0,0,0), d, d
        # 这里简化处理，可以假设光线束RoPE为0，或用更复杂的方式表示
        # 论文中没有明确说明光线束RoPE的计算，但其相对位置很重要
        # 一个简单但有效的做法是，不为光线束应用RoPE，因为它们的相对位置已经由其方向编码
        ray_rope = torch.zeros_like(context_rope_cam[:, :N_bundles])

        # 2.3 通过Transformer块 (解码器模式)
        for block in self.view_dependent_stage:
            ray_tokens = block(ray_tokens,
                               context=context_tokens,
                               x_rope=ray_rope,
                               context_rope=context_rope_cam)

        # 3. 输出头
        pixel_patches = self.output_head(ray_tokens)
        pixel_patches = rearrange(pixel_patches,
                                  'b n (h w c) -> b n h w c',
                                  h=self.patch_size, w=self.patch_size, c=3)

        # 论文提及输出是 log(x+1) 编码的HDR值
        return torch.log1p(torch.relu(pixel_patches))  # 使用relu确保非负
