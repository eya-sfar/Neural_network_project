import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(a):
    return 1 - a**2


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def get_activation(name):
    if name == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif name == "relu":
        return relu, relu_derivative
    elif name == "tanh":
        return tanh, tanh_derivative
    elif name == "softmax":
        return softmax, None
    else:
        raise ValueError(f"Activation inconnue: {name}")