from models.mlp import MLP
from training.trainer import Trainer
from utils.metrics import accuracy


def run_experiment_2(
    X_train,
    X_test,
    y_train,
    y_test,
    learning_rate=0.01,
    epochs=100,
    activation="relu",
    architecture_1=[8],
    architecture_2=[16, 8],
    architecture_3=[32, 16, 8]
):
    architectures = {
        "Architecture 1": architecture_1,
        "Architecture 2": architecture_2,
        "Architecture 3": architecture_3
    }

    results = {}

    for name, layers in architectures.items():
        mlp = MLP(
            input_size=X_train.shape[1],
            hidden_layers=layers,
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

        y_pred = mlp.predict(X_test)

        results[name] = {
            "architecture": layers,
            "accuracy": accuracy(y_test, y_pred),
            "history": history
        }

    return results