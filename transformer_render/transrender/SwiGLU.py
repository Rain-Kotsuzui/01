import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def forward(self, x):
        
        assert x.shape[-1] % 2 == 0, "输入维度需为偶数"
        
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x