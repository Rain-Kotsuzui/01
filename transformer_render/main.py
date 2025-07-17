import transrender.render as render
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    # 模型参数 (遵循论文)
    B = 2  # Batch size
    N_TRI = 1024  # Number of triangles
    N_BUNDLES = 64 * 64 // (8 * 8)  # 64x64 image, 8x8 patches

    # 1. 配置
    checkpoint_path = "checkpoints/step_1000.pt"  # <<<--- 指定你训练好的模型权重
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_image_path = "rendered_image.png"

    # 2. 实例化模型架构 (参数必须和训练时完全一致！)
    config = {
        'd_model': 768,
        'n_heads': 8,  # 确保这里的参数和训练时一致
        'n_layers_indep': 12,
        'n_layers_dep': 6,
        'n_register_tokens': 16,
        'patch_size': 8
    }
    model = render.TransRender(**config).to(device)

    # 3. 加载权重
    print(f"正在从 {checkpoint_path} 加载权重...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model_state_dict']

    # **处理可能存在的'module.'前缀**
    # 这是一个非常稳健的做法，无论保存时模型是否被DDP包装，都能正确加载
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v  # 去掉 'module.'
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    print("权重加载成功！")

    # 4. 设置为评估模式
    model.eval()

    # 1. 伪造三角面片数据
    triangles_data = {
        'vertices': torch.randn(B, N_TRI, 3, 3),
        'vertex_normals': F.normalize(torch.randn(B, N_TRI, 3, 3), dim=-1),
        'brdf_params': torch.rand(B, N_TRI, 7),  # 漫反射(3)+高光(3)+粗糙度(1)
        'emission': torch.rand(B, N_TRI, 3)
    }

    # 2. 伪造相机数据
    camera_data = {
        'ray_bundles_directions': F.normalize(torch.randn(B, N_BUNDLES, 8, 8, 3), dim=-1),
        'world_to_cam_matrix': torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    }

    # 3. 前向传播
    # 在实际使用中，需要将所有数据移动到GPU
    # device = 'cuda'
    # model.to(device)
    # triangles_data = {k: v.to(device) for k, v in triangles_data.items()}
    # camera_data = {k: v.to(device) for k, v in camera_data.items()}

    with torch.no_grad():  # 推理时不需要计算梯度
        print("正在渲染图像...")
        output_patches = model(triangles_data, camera_data)

    print(f"模型成功运行!")
    print(f"输出形状: {output_patches.shape}")
    # 期望输出: [B, N_bundles, patch_size, patch_size, 3]
    # e.g., [2, 64, 8, 8, 3]
