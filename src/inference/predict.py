import sys
import os
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.tokenizer import tokenize_file
from src.data.dataset import chunk_sequence
from src.model.image_generator import generate_images_for_file
from src.model.classifier import images_dict_to_tensor


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_models(config):
    """Load the trained transformer and CNN classifier from disk."""
    transformer_path = Path(config["saved_models"]["transformer"]) / "best_model.h5"
    classifier_path = Path(config["saved_models"]["classifier"]) / "best_model.h5"

    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer model not found: {transformer_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {classifier_path}")

    transformer = tf.keras.models.load_model(str(transformer_path))
    classifier = tf.keras.models.load_model(str(classifier_path))

    return transformer, classifier


def apply_decision(score, config):
    """Map a probability score to a human-readable decision."""
    if score >= config["thresholds"]["block"]:
        return "BLOCKED"
    elif score >= config["thresholds"]["quarantine"]:
        return "QUARANTINE"
    else:
        return "ALLOWED"


def predict(file_path, config, transformer, classifier):
    """
    Full inference pipeline for a single file.
    Returns the malware probability score and the decision.
    """
    file_path = Path(file_path)
    print(f"Analyzing: {file_path.name}")

    # Step 1 — Tokenize
    tokens = tokenize_file(file_path)
    print(f"  Tokens     : {len(tokens):,}")

    # Step 2 — Chunk
    max_seq_len = config["transformer"]["max_seq_len"]
    pad_token_id = config["transformer"]["pad_token_id"]
    chunks = chunk_sequence(tokens, max_seq_len, pad_token_id)
    print(f"  Chunks     : {len(chunks)}")

    # Step 3 — Generate attention images (48 images via RMS aggregation)
    images_dict = generate_images_for_file(transformer, chunks, config)
    print(f"  Images     : {len(images_dict)} (layers x heads)")

    # Step 4 — Convert to tensor
    num_layers = config["transformer"]["num_layers"]
    num_heads = config["transformer"]["num_heads"]
    tensor = images_dict_to_tensor(images_dict, num_layers, num_heads)
    X = np.expand_dims(tensor, axis=0).astype(np.float32)

    # Step 5 — Classify
    score = float(classifier.predict(X, verbose=0).flatten()[0])
    decision = apply_decision(score, config)

    print(f"  Score      : {score:.4f}")
    print(f"  Decision   : {decision}")

    return score, decision


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MalViT — APK malware detector")
    parser.add_argument("file", type=str, help="Path to the APK file to analyze")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    transformer, classifier = load_models(config)
    score, decision = predict(args.file, config, transformer, classifier)
