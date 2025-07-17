import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import lpips
from einops import rearrange

# 假设之前的模型代码保存在 renderformer_model.py
from transrender import render

# --- 1. 配置 ---
class TrainingConfig:
    # 数据和模型参数
    image_size = 256  # 初始训练分辨率
    patch_size = 8
    n_triangles = 1536 # 初始训练三角面片数
    d_model = 768
    n_heads = 8
    n_layers_indep = 12
    n_layers_dep = 6
    n_register_tokens = 16

    # 训练参数
    batch_size = 1  # 这是每个GPU的batch size, 总batch size = batch_size * num_gpus
    num_epochs = 10 # 示例epoch数
    learning_rate = 1e-4
    warmup_steps = 8000
    lpips_weight = 0.05
    
    # 运行时参数
    num_workers = 4
    save_interval = 1000 # 每1000步保存一次checkpoint
    log_interval = 50   # 每50步打印一次日志
    checkpoint_dir = "./checkpoints"



# --- 2. 伪数据加载器 (请用您的真实数据替换这里) ---
class FakeRenderDataset(Dataset):
    """
    一个生成随机伪数据的Dataset。
    在真实场景中，您需要加载由Blender等渲染器生成的数据。
    """
    def __init__(self, size=10000, n_triangles=1536, image_size=256, patch_size=8):
        self.size = size
        self.n_triangles = n_triangles
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_bundles = (image_size // patch_size) ** 2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 伪造三角面片数据
        triangles_data = {
            'vertices': torch.randn(self.n_triangles, 3, 3),
            'vertex_normals': F.normalize(torch.randn(self.n_triangles, 3, 3), dim=-1),
            'brdf_params': torch.rand(self.n_triangles, 7),
            'emission': torch.rand(self.n_triangles, 3)
        }
        
        # 伪造相机数据
        camera_data = {
            'ray_bundles_directions': F.normalize(torch.randn(self.n_bundles, self.patch_size, self.patch_size, 3), dim=-1),
            'world_to_cam_matrix': torch.eye(4)
        }
        
        # 伪造目标图像 (HDR, 值域 > 1)
        # 真实数据应来自渲染器
        target_image_patches = torch.rand(self.n_bundles, self.patch_size, self.patch_size, 3) * 5.0
        
        return triangles_data, camera_data, target_image_patches





# --- 3. 损失函数 (遵循论文) ---
class RenderFormerLoss(nn.Module):
    def __init__(self, lpips_weight=0.05, device='cpu'):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def tone_map(self, x):
        # 论文中的色调映射: clamp(log(I+1)/log(2), 0, 1)
        # x 已经是 log(I+1)
        return torch.clamp(x / math.log(2), 0, 1)

    def forward(self, pred_log, target_hdr):
        # pred_log 的形状: [B, N_bundles, H_patch, W_patch, C]
        # target_hdr 的形状: [B, N_bundles, H_patch, W_patch, C]
        
        # L1 Loss on log-transformed values
        target_log = torch.log1p(target_hdr)
        loss_l1 = F.l1_loss(pred_log, target_log)
        
        # LPIPS Loss
        if self.lpips_weight > 0:
            # 1. 色调映射
            pred_tm = self.tone_map(pred_log)
            target_tm = self.tone_map(target_log)
            
            # 2. 调整形状和范围以适应LPIPS: [B, C, H, W] 和 [-1, 1]
            # 我们将所有patch拼接成一个大图
            B, N_b, H_p, W_p, C = pred_tm.shape
            img_size_sqrt = int(math.sqrt(N_b))
            
            pred_img = rearrange(pred_tm, 'b (h w) hp wp c -> b c (h hp) (w wp)', h=img_size_sqrt, w=img_size_sqrt)
            target_img = rearrange(target_tm, 'b (h w) hp wp c -> b c (h hp) (w wp)', h=img_size_sqrt, w=img_size_sqrt)

            pred_lpips_in = pred_img * 2.0 - 1.0
            target_lpips_in = target_img * 2.0 - 1.0

            loss_lpips = self.lpips_fn(pred_lpips_in, target_lpips_in).mean()
        else:
            loss_lpips = torch.tensor(0.0, device=pred_log.device)
            
        total_loss = loss_l1 + self.lpips_weight * loss_lpips
        return total_loss, loss_l1, loss_lpips

# --- 4. DDP 设置 ---
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

# --- 5. 学习率调度器 ---
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- 6. 训练主函数 ---
def train(rank, world_size, config):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)

    # 模型
    model = render.TransRender(
        d_model=config.d_model, n_heads=config.n_heads, 
        n_layers_indep=config.n_layers_indep, n_layers_dep=config.n_layers_dep,
        n_register_tokens=config.n_register_tokens, patch_size=config.patch_size
    ).to(rank)
    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank], find_unused_parameters=True) # 设为True如果模型有不参与loss计算的参数

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)



    # 数据
    dataset = FakeRenderDataset(size=100000, n_triangles=config.n_triangles, image_size=config.image_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers, pin_memory=True)



    # 学习率调度器和损失函数
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)
    loss_fn = RenderFormerLoss(lpips_weight=config.lpips_weight, device=rank)

    if rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f"开始训练，总步数: {total_steps}")

    # 训练循环
    global_step = 0
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)
        
        # 在主进程上显示进度条
        pbar = tqdm(dataloader, disable=(rank != 0))
        for triangles_data, camera_data, target_patches in pbar:
            # 数据移动到GPU
            triangles_data = {k: v.to(rank) for k, v in triangles_data.items()}
            camera_data = {k: v.to(rank) for k, v in camera_data.items()}
            target_patches = target_patches.to(rank)

            optimizer.zero_grad()
            
            # 前向传播
            # 论文提到可以应用随机旋转作为数据增强
            # 这里我们省略了这一步，但在真实训练中很重要
            pred_patches_log = model(triangles_data, camera_data)

            # 计算损失
            loss, l1, lpips_loss = loss_fn(pred_patches_log, target_patches)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            # 日志和保存
            if rank == 0:
                pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, L1: {l1.item():.4f}, LPIPS: {lpips_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                
                if global_step % config.save_interval == 0:
                    checkpoint_path = os.path.join(config.checkpoint_dir, f"step_{global_step}.pt")
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpoint_path)
                    print(f"在步数 {global_step} 保存checkpoint到 {checkpoint_path}")

    cleanup_ddp()

# --- 7. 启动器 ---
if __name__ == "__main__":
    # 使用 torchrun 启动
    # 例如: torchrun --nproc_per_node=4 train.py
    
    config = TrainingConfig()
    
    # DDP 会自动设置这些环境变量
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        train(rank, world_size, config)
    else:
        # 单GPU训练（为了调试）
        print("以单GPU模式运行...")
        # 伪造DDP环境
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        train(0, 1, config)