import numpy as np


def l2_penalty(parameters, l2_lambda, m):
    """
    Compute L2 regularization penalty.
    Only weights W are regularized, not biases b.
    """
    if l2_lambda <= 0:
        return 0

    penalty = 0

    for key, value in parameters.items():
        if key.startswith("W"):
            penalty += np.sum(np.square(value))

    return (l2_lambda / (2 * m)) * penalty


def add_l2_to_gradients(parameters, gradients, l2_lambda, m):
    """
    Add L2 derivative to weight gradients.
    dW = dW + (lambda / m) * W
    """
    if l2_lambda <= 0:
        return gradients

    for key in parameters.keys():
        if key.startswith("W"):
            layer_number = key[1:]
            grad_key = "dW" + layer_number

            if grad_key in gradients:
                gradients[grad_key] += (l2_lambda / m) * parameters[key]

    return gradients


def apply_dropout(A, dropout_rate):
    """
    Apply inverted dropout during training.
    """
    if dropout_rate <= 0:
        return A, None

    if dropout_rate >= 1:
        raise ValueError("Dropout rate must be less than 1.")

    mask = np.random.rand(*A.shape) > dropout_rate
    A_dropout = A * mask
    A_dropout = A_dropout / (1 - dropout_rate)

    return A_dropout, mask


def apply_dropout_backward(dA, mask, dropout_rate):
    """
    Apply dropout mask during backpropagation.
    """
    if dropout_rate <= 0 or mask is None:
        return dA

    dA = dA * mask
    dA = dA / (1 - dropout_rate)

    return dA


class EarlyStopping:
    """
    Stops training when validation/test loss stops improving.
    """

    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def update(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop