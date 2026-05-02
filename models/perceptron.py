import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_history = []

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            total_errors = 0

            for i in range(n_samples):
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(z)

                error = y[i] - y_pred

                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

                total_errors += abs(error)

            self.errors_history.append(total_errors)

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
from utils.activation import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative, softmax 
from utils.loss import get_loss
class PerceptronActivation:
    def __init__(
        self,
        learning_rate=0.01,
        epochs=100,
        activation="sigmoid",
        loss="binary_cross_entropy"
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_name = activation
        self.loss_name = loss

        self.weights = None
        self.bias = None
        self.loss_history = []

        self._set_activation()
        self._set_loss()

    def _set_activation(self):
        if self.activation_name == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif self.activation_name == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif self.activation_name == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError(f"Activation non supportée: {self.activation_name}")

    def _set_loss(self):
        self.loss_function, self.loss_derivative = get_loss(self.loss_name)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            # Forward
            z = np.dot(X, self.weights) + self.bias
            a = self.activation(z)

            # Loss
            loss = self.loss_function(y, a)
            self.loss_history.append(loss)

            # Backward
            dL_da = self.loss_derivative(y, a)
            da_dz = self.activation_derivative(a)

            dL_dz = dL_da * da_dz

            dW = np.dot(X.T, dL_dz)
            db = np.sum(dL_dz)

            # Update
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)