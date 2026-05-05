import numpy as np
import pandas as pd


def load_csv(file):

    return pd.read_csv(file)

def split_features_target(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y



def handle_missing_values(X):

    X = X.copy()

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].fillna(X[column].mean())
        else:
            X[column] = X[column].fillna(X[column].mode()[0])

    return X




def encode_features(X):

    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded


def normalize_features(X):

    X = X.astype(float)

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std = std.replace(0, 1)

    X_scaled = (X - mean) / std

    return X_scaled, mean, std


def encode_binary_target(y):

    unique_values = sorted(y.unique())

    if len(unique_values) != 2:
        raise ValueError("The target must contain exactly 2 unique values for binary classification.")

    mapping = {
        unique_values[0]: 0,
        unique_values[1]: 1
    }

    y_encoded = y.map(mapping).values

    return y_encoded, mapping


def one_hot_encode_target(y):

    unique_values = sorted(y.unique())

    mapping = {
        value: index for index, value in enumerate(unique_values)
    }

    y_indices = y.map(mapping).values

    y_one_hot = np.zeros((len(y_indices), len(unique_values)))
    y_one_hot[np.arange(len(y_indices)), y_indices] = 1

    return y_one_hot, mapping


def prepare_target(y, problem_type):


    if problem_type == "binary_classification":
        y_prepared, target_mapping = encode_binary_target(y)

        output_size = 1
        output_activation = "sigmoid"
        loss = "binary_cross_entropy"

    elif problem_type == "multiclass_classification":
        y_prepared, target_mapping = one_hot_encode_target(y)

        output_size = len(target_mapping)
        output_activation = "softmax"
        loss = "categorical_cross_entropy"

    elif problem_type == "regression":
        y_prepared = y.values.astype(float)
        target_mapping = None

        output_size = 1
        output_activation = "linear"
        loss = "mse"

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    target_config = {
        "output_size": output_size,
        "output_activation": output_activation,
        "loss": loss,
        "target_mapping": target_mapping
    }

    return y_prepared, target_config


def prepare_data(df, target_column, problem_type):

    # Split
    X, y = split_features_target(df, target_column)

    # Features preprocessing
    X = handle_missing_values(X)
    X = encode_features(X)

    feature_names = list(X.columns)

    X, feature_mean, feature_std = normalize_features(X)
    X = X.values.astype(float)

    # Target preprocessing
    y, target_config = prepare_target(y, problem_type)

    # Final config for model and UI
    config = {
        "input_size": X.shape[1],
        "output_size": target_config["output_size"],
        "output_activation": target_config["output_activation"],
        "loss": target_config["loss"],
        "target_mapping": target_config["target_mapping"],
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std
    }

    return X, y, config
def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_count = int(n_samples * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test