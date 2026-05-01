import numpy as np
from utils.activation import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative, softmax

x= np.array([-1,0,1])

print("Sigmoid:", sigmoid(x))
print("Sigmoid derivative:", sigmoid_derivative(sigmoid(x)))
print("ReLU:", relu(x))
print("ReLU derivative:", relu_derivative(x))
print("Tanh:", tanh(x))
print("Tanh derivative:", tanh_derivative(tanh(x)))
print("Softmax:", softmax(x.reshape(1, -1)))

import numpy as np
from utils.loss import (
    mse_loss,
    mse_loss_derivative,
    binary_cross_entropy_loss,
    binary_cross_entropy_loss_derivative,
    categorical_cross_entropy_loss,
    categorical_cross_entropy_loss_derivative
)

y_true_reg = np.array([[2], [4], [6]])
y_pred_reg = np.array([[2.5], [3.5], [5.5]])

print("MSE:", mse_loss(y_true_reg, y_pred_reg))
print("MSE derivative:", mse_loss_derivative(y_true_reg, y_pred_reg))

y_true_bin = np.array([[1], [0], [1]])
y_pred_bin = np.array([[0.9], [0.2], [0.7]])

print("BCE:", binary_cross_entropy_loss(y_true_bin, y_pred_bin))
print("BCE derivative:", binary_cross_entropy_loss_derivative(y_true_bin, y_pred_bin))

y_true_multi = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

y_pred_multi = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7]
])

print("CCE:", categorical_cross_entropy_loss(y_true_multi, y_pred_multi))
print("CCE derivative:", categorical_cross_entropy_loss_derivative(y_true_multi, y_pred_multi))
from utils.metrics import *

# Classification test
y_true = np.array([[1], [0], [1], [1]])
y_pred = np.array([[0.9], [0.2], [0.8], [0.4]])

print("Accuracy:", accuracy(y_true, y_pred))
print("Precision:", precision(y_true, y_pred))
print("Recall:", recall(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


# Regression test
y_true_reg = np.array([[2], [4], [6]])
y_pred_reg = np.array([[2.5], [3.5], [5.5]])

print("MSE:", mse(y_true_reg, y_pred_reg))
print("RMSE:", rmse(y_true_reg, y_pred_reg))
print("R2:", r2_score(y_true_reg, y_pred_reg))