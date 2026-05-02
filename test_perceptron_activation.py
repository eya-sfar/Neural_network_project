import numpy as np
from models.perceptron import PerceptronActivation

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

model = PerceptronActivation(
    learning_rate=0.1,
    epochs=1000,
    activation="sigmoid",
    loss="binary_cross_entropy"
)

model.fit(X, y)

print("Probabilities:", model.predict_proba(X).flatten())
print("Predictions:", model.predict(X).flatten())
print("Final loss:", model.loss_history[-1])