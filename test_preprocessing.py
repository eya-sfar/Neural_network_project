import pandas as pd
from utils.preprocessing import prepare_data

df = pd.DataFrame({
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "target": [0, 1, 1, 0]
})

X, y, config = prepare_data(
    df=df,
    target_column="target",
    problem_type="binary_classification"
)

print("X:")
print(X)

print("y:")
print(y)

print("config:")
print(config)