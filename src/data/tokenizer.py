import numpy as np
import yaml
import os
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_vocab_size(token_bits):
    return 2 ** token_bits


def tokenize_file(file_path):
    """Read a binary file and return a sequence of integers (0-255)."""
    file_path = Path(file_path)
    raw_bytes = file_path.read_bytes()
    return np.frombuffer(raw_bytes, dtype=np.uint8)


def tokenize_directory(input_dir, output_dir):
    """Tokenize all files in a directory and save as .npy files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.iterdir())
    print(f"Tokenizing {len(files)} file(s) from {input_dir} ...")

    for file_path in files:
        if file_path.is_file():
            tokens = tokenize_file(file_path)
            output_path = output_dir / (file_path.stem + ".npy")
            np.save(output_path, tokens)
            print(f"  {file_path.name} -> {tokens.shape[0]} tokens -> {output_path.name}")

    print("Tokenization complete.")


if __name__ == "__main__":
    config = load_config()

    VOCAB_SIZE = compute_vocab_size(config["transformer"]["token_bits"])
    print(f"Vocabulary size: {VOCAB_SIZE}")

    tokenize_directory(
        config["data"]["raw_benign"],
        config["data"]["processed_benign"]
    )

    tokenize_directory(
        config["data"]["raw_malware"],
        config["data"]["processed_malware"]
    )
