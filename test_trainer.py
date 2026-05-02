import numpy as np

from models.mlp import MLP
from training.trainer import Trainer


# ============================================================
# 1. Binary classification - XOR
# ============================================================

print("\n=== Binary Classification: XOR ===")

X_binary = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_binary = np.array([0, 1, 1, 0])

model_binary = MLP(
    input_size=2,
    hidden_layers=[4],
    output_size=1,
    learning_rate=0.1,
    epochs=5000,
    activation="tanh",
    output_activation="sigmoid",
    loss="binary_cross_entropy"
)

trainer_binary = Trainer(
    model=model_binary,
    problem_type="binary_classification"
)

history_binary = trainer_binary.train(
    X_binary,
    y_binary,
    test_size=0.25
)

print("Final train loss:", history_binary["train_loss"][-1])
print("Final test loss:", history_binary["test_loss"][-1])
print("Final train accuracy:", history_binary["train_metric"][-1])
print("Final test accuracy:", history_binary["test_metric"][-1])
print("Predictions:", model_binary.predict(X_binary).reshape(-1))
print("True labels:", y_binary)


# ============================================================
# 2. Multiclass classification
# ============================================================

print("\n=== Multiclass Classification ===")

X_multi = np.array([
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
    [1.0, 1.0],
    [0.8, 0.8]
])

y_multi = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
])

model_multi = MLP(
    input_size=2,
    hidden_layers=[8],
    output_size=3,
    learning_rate=0.1,
    epochs=5000,
    activation="tanh",
    output_activation="softmax",
    loss="categorical_cross_entropy"
)

trainer_multi = Trainer(
    model=model_multi,
    problem_type="multiclass_classification"
)

history_multi = trainer_multi.train(
    X_multi,
    y_multi,
    test_size=0.33
)

print("Final train loss:", history_multi["train_loss"][-1])
print("Final test loss:", history_multi["test_loss"][-1])
print("Final train accuracy:", history_multi["train_metric"][-1])
print("Final test accuracy:", history_multi["test_metric"][-1])
print("Predictions:", model_multi.predict(X_multi))
print("True labels:", np.argmax(y_multi, axis=1))


# ============================================================
# 3. Regression
# ============================================================

print("\n=== Regression ===")

X_reg = np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9]
], dtype=float)

y_reg = np.array([
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18
], dtype=float)

model_reg = MLP(
    input_size=1,
    hidden_layers=[8],
    output_size=1,
    learning_rate=0.01,
    epochs=10000,
    activation="relu",
    output_activation="linear",
    loss="mse"
)

trainer_reg = Trainer(
    model=model_reg,
    problem_type="regression"
)

history_reg = trainer_reg.train(
    X_reg,
    y_reg,
    test_size=0.2
)

print("Final train loss:", history_reg["train_loss"][-1])
print("Final test loss:", history_reg["test_loss"][-1])
print("Final train MSE:", history_reg["train_metric"][-1])
print("Final test MSE:", history_reg["test_metric"][-1])
print("Predictions:", model_reg.predict(X_reg).reshape(-1))
print("True values:", y_reg)