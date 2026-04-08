import sys
import os
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.model.classifier import build_cnn, images_dict_to_tensor
from src.model.image_generator import generate_and_save_images


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_images_as_tensors(images_dir, config):
    """
    Load saved attention images from disk and convert to tensors.
    Expects structure: images_dir/benign/ and images_dir/malware/
    Returns X (N, 256, 256, 48) and y (N,) arrays.
    """
    num_layers = config["transformer"]["num_layers"]
    num_heads = config["transformer"]["num_heads"]
    vocab_size = 2 ** config["transformer"]["token_bits"]
    num_channels = num_layers * num_heads

    images_dir = Path(images_dir)
    label_map = {"benign": 0, "malware": 1}

    file_tensors = {}

    for label_name, label in label_map.items():
        label_dir = images_dir / label_name
        if not label_dir.exists():
            continue

        png_files = sorted(label_dir.glob("*.png"))
        file_indices = sorted(set(
            int(f.stem.split("_")[0].replace("file", ""))
            for f in png_files
        ))

        for file_idx in file_indices:
            images_dict = {}
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    filename = f"file{file_idx:05d}_layer{layer_idx}_head{head_idx}.png"
                    img_path = label_dir / filename
                    if img_path.exists():
                        img = Image.open(img_path)
                        images_dict[(layer_idx, head_idx)] = img

            if len(images_dict) == num_channels:
                tensor = images_dict_to_tensor(images_dict, num_layers, num_heads)
                key = (label_name, file_idx)
                file_tensors[key] = (tensor, label)

    X = np.array([v[0] for v in file_tensors.values()], dtype=np.float32)
    y = np.array([v[1] for v in file_tensors.values()], dtype=np.float32)

    return X, y


def train(config, images_dir):
    X, y = load_images_as_tensors(images_dir, config)
    print(f"Loaded {X.shape[0]} samples, shape {X.shape}")

    from sklearn.model_selection import train_test_split

    min_class_count = np.bincount(y.astype(int)).min()
    use_stratify = min_class_count >= 2

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42,
        stratify=y if use_stratify else None
    )
    min_temp_class = np.bincount(y_temp.astype(int)).min() if len(np.unique(y_temp)) > 1 else 0
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=y_temp if min_temp_class >= 2 else None
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    model = build_cnn(config)

    output_dir = Path(config["saved_models"]["classifier"])
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
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    try:
        model.fit(
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
        return model

    model.save(str(output_dir / "final_model.h5"))
    print(f"Model saved to {output_dir}")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss     : {loss:.4f}")
    print(f"Test accuracy : {accuracy:.4f}")

    return model


if __name__ == "__main__":
    config = load_config()
    images_dir = config["data"]["images_train"].replace("train/", "")
    train(config, images_dir)
