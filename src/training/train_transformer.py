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


def load_or_build_model(config, output_dir):
    """Load existing model if available, otherwise build a new one."""
    from transformers import TFBertModel

    for candidate in ["best_model.h5", "interrupted_model.h5", "final_model.h5"]:
        model_path = output_dir / candidate
        if model_path.exists():
            print(f"Resuming from existing model: {model_path}")
            return tf.keras.models.load_model(
                str(model_path),
                custom_objects={"TFBertModel": TFBertModel}
            )

    print("No existing model found. Building a new model ...")
    return build_model(config)


def train(config):
    X_train, X_val, X_test, y_train, y_val, y_test = build_dataset(config)

    output_dir = Path(config["saved_models"]["transformer"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_or_build_model(config, output_dir)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config["classifier"]["epochs"],
            batch_size=config["classifier"]["batch_size"],
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state ...")
        model.save(str(output_dir / "interrupted_model.h5"))
        print(f"Model saved to {output_dir / 'interrupted_model.h5'}")
        return model, None

    model.save(str(output_dir / "final_model.h5"))
    print(f"Model saved to {output_dir}")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss     : {loss:.4f}")
    print(f"Test accuracy : {accuracy:.4f}")

    return model, history


if __name__ == "__main__":
    config = load_config()
    train(config)
