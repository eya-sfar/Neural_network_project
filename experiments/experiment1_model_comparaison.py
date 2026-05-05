from models.perceptron import Perceptron
from models.mlp import MLP
from training.trainer import Trainer
from utils.metrics import accuracy


def run_experiment_1(
    X_train,
    X_test,
    y_train,
    y_test,
    learning_rate=0.01,
    epochs=100,
    hidden_layers=[8, 4],
    activation="relu"
):
    results = {}

    # Perceptron
    perceptron = Perceptron(
        learning_rate=learning_rate,
        epochs=epochs
    )

    perceptron.fit(X_train, y_train)
    y_pred_perceptron = perceptron.predict(X_test)

    results["Perceptron"] = {
        "accuracy": accuracy(y_test, y_pred_perceptron)
    }

    # MLP
    mlp = MLP(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_size=1,
        activation=activation,
        output_activation="sigmoid",
        learning_rate=learning_rate
    )

    trainer = Trainer(mlp)

    history = trainer.train(
        X_train,
        y_train,
        epochs=epochs
    )

    y_pred_mlp = mlp.predict(X_test)

    results["MLP"] = {
        "accuracy": accuracy(y_test, y_pred_mlp),
        "history": history,
        "architecture": hidden_layers
    }

    return results