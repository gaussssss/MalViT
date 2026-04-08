import sys
import os
import yaml
import tensorflow as tf
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.dataset import load_sequences_per_file
from src.model.image_generator import generate_and_save_images


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config):
    transformer_path = Path(config["saved_models"]["transformer"]) / "best_model.h5"
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer model not found: {transformer_path}. Run train_transformer.py first.")

    from transformers import TFBertModel
    print(f"Loading transformer from {transformer_path} ...")
    transformer = tf.keras.models.load_model(
        str(transformer_path),
        custom_objects={"TFBertModel": TFBertModel}
    )

    max_seq_len = config["transformer"]["max_seq_len"]
    pad_token_id = config["transformer"]["pad_token_id"]

    print("\nLoading benign files ...")
    benign_chunks, benign_labels = load_sequences_per_file(
        config["data"]["processed_benign"], label=0,
        max_seq_len=max_seq_len, pad_token_id=pad_token_id
    )

    print("\nLoading malware files ...")
    malware_chunks, malware_labels = load_sequences_per_file(
        config["data"]["processed_malware"], label=1,
        max_seq_len=max_seq_len, pad_token_id=pad_token_id
    )

    all_chunks = benign_chunks + malware_chunks
    all_labels = benign_labels + malware_labels

    # Sort by number of chunks ascending (smallest files first)
    sorted_pairs = sorted(zip(all_chunks, all_labels), key=lambda x: len(x[0]))
    all_chunks, all_labels = zip(*sorted_pairs)

    print(f"\nGenerating images for {len(all_chunks)} files (sorted by chunk count) ...")
    output_dir = Path(config["data"]["images_train"]).parent

    generate_and_save_images(
        model=transformer,
        files_chunks=all_chunks,
        labels=all_labels,
        output_dir=output_dir,
        config=config
    )


if __name__ == "__main__":
    config = load_config()
    main(config)
