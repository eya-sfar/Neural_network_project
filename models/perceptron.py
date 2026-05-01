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