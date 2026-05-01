import numpy as np
from models.perceptron import Perceptron

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

model = Perceptron(learning_rate=0.1, epochs=10)
model.fit(X, y)

predictions = model.predict(X)

print("Predictions:", predictions)
print("True labels:", y)
print("Weights:", model.weights)
print("Bias:", model.bias)
print("Errors history:", model.errors_history)