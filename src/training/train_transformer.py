import sys
import os
import yaml
import tensorflow as tf
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.dataset import build_dataset
from src.model.transformer import build_model


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config):
    X_train, X_val, X_test, y_train, y_val, y_test = build_dataset(config)

    model = build_model(config)

    output_dir = Path(config["saved_models"]["transformer"])
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["classifier"]["epochs"],
        batch_size=config["classifier"]["batch_size"],
        callbacks=callbacks,
    )

    model.save(str(output_dir / "final_model.h5"))
    print(f"Model saved to {output_dir}")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss     : {loss:.4f}")
    print(f"Test accuracy : {accuracy:.4f}")

    return model, history


if __name__ == "__main__":
    config = load_config()
    train(config)
