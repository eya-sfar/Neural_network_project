import numpy as np

from utils.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    mse
)


class Trainer:
    def __init__(self, model, problem_type="binary_classification"):
        self.model = model
        self.problem_type = problem_type

        self.history = {
            "train_loss": [],
            "test_loss": [],

            "train_accuracy": [],
            "test_accuracy": [],

            "train_precision": [],
            "test_precision": [],

            "train_recall": [],
            "test_recall": [],

            "train_f1": [],
            "test_f1": [],

            "train_metric": [],
            "test_metric": []
        }

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train_test_split(self, X, y, test_size=0.2, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        split_index = int(n_samples * (1 - test_size))

        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        return (
            X[train_indices],
            X[test_indices],
            y[train_indices],
            y[test_indices]
        )

    def _prepare_y_for_model(self, y):
        if self.model.output_size == 1:
            return y.reshape(-1, 1)

        return y

    def _predicted_labels(self, y_pred):
        if self.problem_type == "binary_classification":
            return (y_pred >= 0.5).astype(int).reshape(-1)

        elif self.problem_type == "multiclass_classification":
            return np.argmax(y_pred, axis=1)

        else:
            return y_pred.reshape(-1)

    def _true_labels(self, y_true):
        if self.problem_type == "binary_classification":
            return y_true.reshape(-1).astype(int)

        elif self.problem_type == "multiclass_classification":
            return np.argmax(y_true, axis=1)

        else:
            return y_true.reshape(-1)

    def _compute_and_store_metrics(self, y_train_true, y_train_pred, y_test_true, y_test_pred):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:

            train_true = self._true_labels(y_train_true)
            test_true = self._true_labels(y_test_true)

            train_pred = self._predicted_labels(y_train_pred)
            test_pred = self._predicted_labels(y_test_pred)

            average = "binary" if self.problem_type == "binary_classification" else "macro"

            train_acc = accuracy(train_true, train_pred)
            test_acc = accuracy(test_true, test_pred)

            train_prec = precision(train_true, train_pred, average=average)
            test_prec = precision(test_true, test_pred, average=average)

            train_rec = recall(train_true, train_pred, average=average)
            test_rec = recall(test_true, test_pred, average=average)

            train_f1 = f1_score(train_true, train_pred, average=average)
            test_f1 = f1_score(test_true, test_pred, average=average)

            self.history["train_accuracy"].append(train_acc)
            self.history["test_accuracy"].append(test_acc)

            self.history["train_precision"].append(train_prec)
            self.history["test_precision"].append(test_prec)

            self.history["train_recall"].append(train_rec)
            self.history["test_recall"].append(test_rec)

            self.history["train_f1"].append(train_f1)
            self.history["test_f1"].append(test_f1)

            self.history["train_metric"].append(train_acc)
            self.history["test_metric"].append(test_acc)

        elif self.problem_type == "regression":
            train_mse = mse(y_train_true, y_train_pred)
            test_mse = mse(y_test_true, y_test_pred)

            self.history["train_accuracy"].append(None)
            self.history["test_accuracy"].append(None)

            self.history["train_precision"].append(None)
            self.history["test_precision"].append(None)

            self.history["train_recall"].append(None)
            self.history["test_recall"].append(None)

            self.history["train_f1"].append(None)
            self.history["test_f1"].append(None)

            self.history["train_metric"].append(train_mse)
            self.history["test_metric"].append(test_mse)

        else:
            raise ValueError(f"Problem type inconnu: {self.problem_type}")

    def train(self, X, y, test_size=0.2, callback=None):
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(
            X,
            y,
            test_size=test_size
        )

        y_train_model = self._prepare_y_for_model(self.y_train)
        y_test_model = self._prepare_y_for_model(self.y_test)

        for epoch in range(self.model.epochs):
            y_train_pred, cache = self.model.forward(self.X_train)

            train_loss = self.model.loss_function(y_train_model, y_train_pred)

            gradients = self.model.backward(y_train_model, cache)
            self.model.update_parameters(gradients)

            y_test_pred, _ = self.model.forward(self.X_test)
            test_loss = self.model.loss_function(y_test_model, y_test_pred)

            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)

            self._compute_and_store_metrics(
                y_train_model,
                y_train_pred,
                y_test_model,
                y_test_pred
            )

            if callback is not None:
                callback(epoch, self.history)

        return self.history