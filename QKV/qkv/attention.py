import numpy as np
from .softmax import softmax


def qkv_attention(Q, K, V, mask=None):
    Q = np.array(Q)
    K = np.array(K)
    V = np.array(V)
    
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    scaled_scores = scores / np.sqrt(d_k)

    if mask is not None:
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)
    attention_weights = softmax(scaled_scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights
