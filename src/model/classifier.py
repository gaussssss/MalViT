import numpy as np
import yaml
import tensorflow as tf
from pathlib import Path


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def images_dict_to_tensor(images_dict, num_layers, num_heads):
    """
    Convert a {(layer, head): PIL Image} dict to a (256, 256, num_layers*num_heads) tensor.
    Images are stacked in order: layer 0 head 0, layer 0 head 1, ..., layer L head H.
    """
    channels = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            img = images_dict[(layer_idx, head_idx)]
            channel = np.array(img, dtype=np.float32) / 255.0
            channels.append(channel)
    return np.stack(channels, axis=-1)


def build_cnn(config):
    """Build and compile the CNN binary classifier."""
    num_layers = config["transformer"]["num_layers"]
    num_heads = config["transformer"]["num_heads"]
    num_channels = num_layers * num_heads
    vocab_size = 2 ** config["transformer"]["token_bits"]

    inputs = tf.keras.Input(shape=(vocab_size, vocab_size, num_channels), name="attention_images")

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["classifier"]["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    config = load_config()
    model = build_cnn(config)
    model.summary()
