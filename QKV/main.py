from qkv import attention
import numpy as np

batch_size = 1
seq_len = 4  # 序列长度为 4
d_k = 8      # Key/Query 向量维度
d_v = 16     # Value 向量维度

# 随机生成 Q, K, V 矩阵
np.random.seed(42)  # 为了结果可复现
Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_v)


def test1():

    # --- 场景1: 不使用掩码 ---
    print("--- 场景1: 不使用掩码 ---")
    output_no_mask, weights_no_mask = attention.qkv_attention(Q, K, V)

    print("Output (无掩码) 的形状:", output_no_mask.shape)
    print("Attention Weights (无掩码) 的形状:", weights_no_mask.shape)
    print("\n注意力权重矩阵 (无掩码) [第一个样本]:")
    print(weights_no_mask[0])
# 验证权重和为 1
    print("\n每行的权重和 (应接近1):", np.sum(weights_no_mask[0], axis=-1))
    pass


def test2():
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
# 扩展到 batch_size
# shape (1, seq_len, seq_len)
    causal_mask = np.expand_dims(causal_mask, axis=0)

    print("\n\n--- 场景2: 使用因果掩码 ---")
    output_with_mask, weights_with_mask = attention.qkv_attention(
        Q, K, V, mask=causal_mask)

    print("Output (有掩码) 的形状:", output_with_mask.shape)
    print("Attention Weights (有掩码) 的形状:", weights_with_mask.shape)
    print("\n注意力权重矩阵 (有掩码) [第一个样本]:")
# 权重矩阵的上三角部分应该为0
    print(weights_with_mask[0])
# 验证权重和为 1
    print("\n每行的权重和 (应接近1):", np.sum(weights_with_mask[0], axis=-1))
    pass


if __name__ == "__main__":
    test1()
    test2()
