import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def chunk_sequence(sequence, max_seq_len, pad_token_id):
    """Split a sequence into fixed-length chunks, padding the last one if needed."""
    chunks = []
    for start in range(0, len(sequence), max_seq_len):
        chunk = sequence[start:start + max_seq_len]
        if len(chunk) < max_seq_len:
            padding = np.full(max_seq_len - len(chunk), pad_token_id, dtype=np.int32)
            chunk = np.concatenate([chunk, padding])
        chunks.append(chunk.astype(np.int32))
    return chunks


def load_sequences(processed_dir, label, max_seq_len, pad_token_id):
    """Load all .npy files from a directory, chunk them, and assign a label."""
    processed_dir = Path(processed_dir)
    all_chunks = []
    all_labels = []

    for npy_file in sorted(processed_dir.glob("*.npy")):
        sequence = np.load(npy_file)
        if len(sequence) == 0:
            continue
        chunks = chunk_sequence(sequence, max_seq_len, pad_token_id)
        all_chunks.extend(chunks)
        all_labels.extend([label] * len(chunks))
        print(f"  {npy_file.name} -> {len(chunks)} chunk(s)")

    return all_chunks, all_labels


def load_sequences_per_file(processed_dir, label, max_seq_len, pad_token_id):
    """Load all .npy files from a directory and return chunks grouped by file."""
    processed_dir = Path(processed_dir)
    files_chunks = []
    labels = []

    for npy_file in sorted(processed_dir.glob("*.npy")):
        sequence = np.load(npy_file)
        if len(sequence) == 0:
            continue
        chunks = chunk_sequence(sequence, max_seq_len, pad_token_id)
        files_chunks.append(chunks)
        labels.append(label)
        print(f"  {npy_file.name} -> {len(chunks)} chunk(s)")

    return files_chunks, labels


def build_dataset(config, val_size=0.15, test_size=0.15, random_state=42):
    """Load, chunk, and split data into train/val/test sets."""
    max_seq_len = config["transformer"]["max_seq_len"]
    pad_token_id = config["transformer"]["pad_token_id"]

    print("Loading benign files ...")
    benign_chunks, benign_labels = load_sequences(
        config["data"]["processed_benign"], label=0, max_seq_len=max_seq_len, pad_token_id=pad_token_id
    )

    print("Loading malware files ...")
    malware_chunks, malware_labels = load_sequences(
        config["data"]["processed_malware"], label=1, max_seq_len=max_seq_len, pad_token_id=pad_token_id
    )

    X = np.array(benign_chunks + malware_chunks, dtype=np.int32)
    y = np.array(benign_labels + malware_labels, dtype=np.int32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )

    print(f"\nDataset summary:")
    print(f"  Train : {X_train.shape[0]} chunks")
    print(f"  Val   : {X_val.shape[0]} chunks")
    print(f"  Test  : {X_test.shape[0]} chunks")
    print(f"  Shape : {X_train.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    config = load_config()
    X_train, X_val, X_test, y_train, y_val, y_test = build_dataset(config)
