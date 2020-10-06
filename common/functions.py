# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """激活函数sigmoid，一般用于二元分类问题

    Args:
        x (array): numpy数组

    Returns:
        array: 计算后的结果
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    """激活函数relu

    Args:
        x (array): 数组

    Returns:
        array: 激活后的输出值
    """
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    """激活函数，一般用于分类问题

    Args:
        x (array): 数组

    Returns:
        array: 输出
    """
    #当输入多个样本时，
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    """损失函数均方误差

    Args:
        y (array): 神经网络的输出
        t (array): 监督数据

    Returns:
        [type]: [description]
    """
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    """计算交叉熵误差

    Args:
        y ([type]): [description]
        t ([type]): [description]

    Returns:
        [type]: [description]
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    result = sigmoid(x)
    print(result)
