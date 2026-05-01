import numpy as np


def mse_loss(y_true,y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_loss_derivative(y_true,y_pred):
    return 2*(y_pred - y_true) / y_true.shape[0]

def binary_cross_entropy_loss(y_true,y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def binary_cross_entropy_loss_derivative(y_true,y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon , 1-epsilon)
    return (y_pred - y_true) / (y_pred * (1-y_pred)*y_true.shape[0])


def categorical_cross_entropy_loss(y_true,y_pred):
    epsilon = 1e-15
    y_pred =np.clip(y_pred, epsilon, 1-epsilon)
    return np.mean(np.sum(-y_true * np.log(y_pred), axis=1))

def categorical_cross_entropy_loss_derivative(y_true,y_pred): 
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon , 1-epsilon)
    return -y_true / y_pred / y_true.shape[0]
def get_loss(name):
    if name == "mse":
        return mse_loss, mse_loss_derivative
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_loss, binary_cross_entropy_loss_derivative
    elif name == "categorical_cross_entropy":
        return categorical_cross_entropy_loss, categorical_cross_entropy_loss_derivative
    else:
        raise ValueError(f"Loss inconnue: {name}")


