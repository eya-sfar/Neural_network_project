import numpy as np
import matplotlib.pyplot as plt

from models.mlp import MLP
from training.trainer import Trainer
from utils.visualization import (
    plot_loss,
    plot_metric,
    plot_decision_boundary
)

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
    epochs=5000,
    activation="tanh",
    output_activation="sigmoid",
    loss="binary_cross_entropy"
)

trainer = Trainer(model, problem_type="binary_classification")
history = trainer.train(X, y, test_size=0.25)

fig1 = plot_loss(history)
fig2 = plot_metric(history, "binary_classification")
fig3 = plot_decision_boundary(model, X, y)

plt.show()