import numpy as np
import matplotlib.pyplot as plt




def plot_loss(history):
    """
    Affiche Train Loss et Test Loss.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)

    return plt.gcf()



def plot_metric(history, problem_type):
    """
    Affiche Accuracy pour classification ou MSE pour regression.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_metric"], label="Train Metric")
    plt.plot(history["test_metric"], label="Test Metric")

    plt.xlabel("Epochs")

    if problem_type in ["binary_classification", "multiclass_classification"]:
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Epochs")
    elif problem_type == "regression":
        plt.ylabel("MSE")
        plt.title("MSE vs Epochs")
    else:
        plt.ylabel("Metric")
        plt.title("Metric vs Epochs")

    plt.legend()
    plt.grid(True)

    return plt.gcf()




def plot_decision_boundary(model, X, y):
    """
    Affiche la frontière de décision pour un dataset 2D.

    Fonctionne pour :
    - binary classification
    - multiclass classification
    """

    if X.shape[1] != 2:
        raise ValueError("Decision boundary disponible seulement pour 2 features.")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = model.predict(grid)

    if predictions.ndim > 1:
        predictions = predictions.flatten()

    Z = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y if y.ndim == 1 else np.argmax(y, axis=1), edgecolors="k")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.grid(True)

    return plt.gcf()



def plot_confusion_matrix(cm):
    """
    Affiche une matrice de confusion.
    """
    plt.figure(figsize=(5, 4))

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()

    return plt.gcf()



def plot_predicted_vs_actual(y_true, y_pred):
    """
    Affiche y réel vs y prédit pour régression.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    plt.figure(figsize=(6, 6))

    plt.scatter(y_true, y_pred)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.plot([min_value, max_value], [min_value, max_value])

    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Predicted vs Actual")
    plt.grid(True)

    return plt.gcf()


def plot_residuals(y_true, y_pred):
    """
    Affiche les résidus pour régression.
    Résidu = y_true - y_pred
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(y_pred, residuals)
    plt.axhline(0)

    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)

    return plt.gcf()