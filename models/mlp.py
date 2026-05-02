import numpy as np

from utils.activation import get_activation
from utils.loss import get_loss


class MLP:
    def __init__(
        self,
        input_size,
        hidden_layers=[4],
        output_size=1,
        learning_rate=0.01,
        epochs=1000,
        activation="relu",
        output_activation="sigmoid",
        loss="binary_cross_entropy"
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.activation_name = activation
        self.output_activation_name = output_activation
        self.loss_name = loss

        self.activation, self.activation_derivative = get_activation(activation)
        self.output_activation, self.output_activation_derivative = get_activation(output_activation)

        self.loss_function, self.loss_derivative = get_loss(loss)

        self.parameters = {}
        self.loss_history = []

        self._initialize_parameters()

    def _initialize_parameters(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        for layer in range(1, len(layer_sizes)):
            self.parameters[f"W{layer}"] = np.random.randn(
                layer_sizes[layer - 1],
                layer_sizes[layer]
            ) * np.sqrt(2 / layer_sizes[layer - 1])

            self.parameters[f"b{layer}"] = np.zeros((1, layer_sizes[layer]))

    def _activation_derivative_value(self, Z, A):
        if self.activation_name == "relu":
            return self.activation_derivative(Z)

        elif self.activation_name in ["sigmoid", "tanh"]:
            return self.activation_derivative(A)

        else:
            raise ValueError(f"Activation inconnue: {self.activation_name}")

    def forward(self, X):
        cache = {}

        A = X
        cache["A0"] = X

        number_of_layers = len(self.hidden_layers) + 1

        for layer in range(1, number_of_layers):
            W = self.parameters[f"W{layer}"]
            b = self.parameters[f"b{layer}"]

            Z = np.dot(A, W) + b
            A = self.activation(Z)

            cache[f"Z{layer}"] = Z
            cache[f"A{layer}"] = A

        W_output = self.parameters[f"W{number_of_layers}"]
        b_output = self.parameters[f"b{number_of_layers}"]

        Z_output = np.dot(A, W_output) + b_output
        A_output = self.output_activation(Z_output)

        cache[f"Z{number_of_layers}"] = Z_output
        cache[f"A{number_of_layers}"] = A_output

        return A_output, cache

    def backward(self, y, cache):
        gradients = {}

        m = y.shape[0]
        number_of_layers = len(self.hidden_layers) + 1

        A_output = cache[f"A{number_of_layers}"]

        if self.output_size == 1:
            y = y.reshape(-1, 1)

        # Cas classiques simplifiés :
        # sigmoid + binary_cross_entropy
        # softmax + categorical_cross_entropy
        # linear/mse plus tard si tu ajoutes linear
        if self.loss_name in ["binary_cross_entropy", "categorical_cross_entropy"]:
            dZ = A_output - y
        else:
            dA = self.loss_derivative(y, A_output)

            if self.output_activation_name == "sigmoid":
                dZ = dA * self.output_activation_derivative(A_output)
            else:
                dZ = dA

        for layer in reversed(range(1, number_of_layers + 1)):
            A_previous = cache[f"A{layer - 1}"]

            gradients[f"dW{layer}"] = np.dot(A_previous.T, dZ) / m
            gradients[f"db{layer}"] = np.sum(dZ, axis=0, keepdims=True) / m

            if layer > 1:
                W = self.parameters[f"W{layer}"]
                dA_previous = np.dot(dZ, W.T)

                Z_previous = cache[f"Z{layer - 1}"]
                A_previous_hidden = cache[f"A{layer - 1}"]

                dZ = dA_previous * self._activation_derivative_value(
                    Z_previous,
                    A_previous_hidden
                )

        return gradients

    def update_parameters(self, gradients):
        number_of_layers = len(self.hidden_layers) + 1

        for layer in range(1, number_of_layers + 1):
            self.parameters[f"W{layer}"] -= self.learning_rate * gradients[f"dW{layer}"]
            self.parameters[f"b{layer}"] -= self.learning_rate * gradients[f"db{layer}"]

    def fit(self, X, y):
        if self.output_size == 1:
            y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            y_pred, cache = self.forward(X)

            loss = self.loss_function(y, y_pred)
            self.loss_history.append(loss)

            gradients = self.backward(y, cache)
            self.update_parameters(gradients)

    def predict_proba(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba(X)

        if self.output_activation_name == "sigmoid":
            return (y_pred >= 0.5).astype(int)

        elif self.output_activation_name == "softmax":
            return np.argmax(y_pred, axis=1)

        else:
            return y_pred