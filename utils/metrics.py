import numpy as np


def flatten(y):
    return np.asarray(y).reshape(-1)


def accuracy(y_true, y_pred):
    y_true = flatten(y_true).astype(int)
    y_pred = flatten(y_pred).astype(int)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    y_true = flatten(y_true).astype(int)
    y_pred = flatten(y_pred).astype(int)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    label_to_index = {label: i for i, label in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        cm[label_to_index[t], label_to_index[p]] += 1

    return cm, labels


def precision(y_true, y_pred, average="binary"):
    cm, labels = confusion_matrix(y_true, y_pred)

    precisions = []

    for i in range(len(labels)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precisions.append(tp / (tp + fp + 1e-15))

    if average == "binary":
        if len(precisions) < 2:
            return precisions[0]
        return precisions[1]

    return np.mean(precisions)


def recall(y_true, y_pred, average="binary"):
    cm, labels = confusion_matrix(y_true, y_pred)

    recalls = []

    for i in range(len(labels)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recalls.append(tp / (tp + fn + 1e-15))

    if average == "binary":
        if len(recalls) < 2:
            return recalls[0]
        return recalls[1]

    return np.mean(recalls)


def f1_score(y_true, y_pred, average="binary"):
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)

    return 2 * p * r / (p + r + 1e-15)


def classification_metrics(y_true, y_pred, average="binary"):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, average=average),
        "recall": recall(y_true, y_pred, average=average),
        "f1_score": f1_score(y_true, y_pred, average=average)
    }


def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    return 1 - (ss_residual / (ss_total + 1e-15))