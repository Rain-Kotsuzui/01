import numpy as np


def softmax(x, axis=-1):
    exp_x=np.exp(x)
    exp_sum = np.sum(exp_x, axis=axis, keepdims=True)
    result = exp_x / exp_sum
    return result
