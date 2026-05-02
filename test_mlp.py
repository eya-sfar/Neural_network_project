import numpy as np
from models.mlp import MLP

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

model = MLP(
    input_size=2,
    hidden_layers=[4],
    output_size=1,
    learning_rate=0.1,
    epochs=10000,
    activation="tanh",
    output_activation="sigmoid",
    loss="binary_cross_entropy"
)

model.fit(X, y)

print("Probabilities:", model.predict_proba(X).flatten())
print("Predictions:", model.predict(X).flatten())
print("True labels:", y)
print("Final loss:", model.loss_history[-1])