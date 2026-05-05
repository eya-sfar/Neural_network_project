from models.mlp import MLP
from training.trainer import Trainer
from utils.metrics import accuracy


def run_experiment_3(
    X_train,
    X_test,
    y_train,
    y_test,
    learning_rate=0.01,
    epochs=300,
    hidden_layers=[32, 16],
    activation="relu",
    l2_lambda=0.01,
    patience=20
):
    results = {}

    configs = {
        "Without Regularization": {
            "l2_lambda": 0.0,
            "early_stopping": False
        },
        "With L2 Regularization": {
            "l2_lambda": l2_lambda,
            "early_stopping": False
        },
        "With Early Stopping": {
            "l2_lambda": 0.0,
            "early_stopping": True
        }
    }

    for name, config in configs.items():
        mlp = MLP(
            input_size=X_train.shape[1],
            hidden_layers=hidden_layers,
            output_size=1,
            activation=activation,
            output_activation="sigmoid",
            learning_rate=learning_rate,
            l2_lambda=config["l2_lambda"]
        )

        trainer = Trainer(mlp)

        history = trainer.train(
            X_train,
            y_train,
            epochs=epochs,
            early_stopping=config["early_stopping"],
            patience=patience
        )

        y_pred = mlp.predict(X_test)

        results[name] = {
            "accuracy": accuracy(y_test, y_pred),
            "history": history,
            "l2_lambda": config["l2_lambda"],
            "early_stopping": config["early_stopping"]
        }

    return results