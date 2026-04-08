import numpy as np
import yaml
from pathlib import Path
from PIL import Image


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_indicator(tokens, vocab_size):
    """
    Build indicator matrix (vocab_size, seq_len) using broadcasting.
    indicator[v, p] = 1 if tokens[p] == v, else 0.
    """
    return (np.arange(vocab_size)[:, None] == tokens[None, :]).astype(np.float32)


def aggregate_attention_to_vocab(attention_matrix, input_tokens, vocab_size=256):
    """
    Aggregate a (seq_len, seq_len) attention matrix into a (vocab_size, vocab_size)
    matrix using fully vectorized NumPy operations.
    """
    tokens = np.clip(np.array(input_tokens, dtype=np.int32), 0, vocab_size - 1)

    indicator = build_indicator(tokens, vocab_size)        # (vocab_size, seq_len)
    counts = indicator.sum(axis=1, keepdims=True)          # (vocab_size, 1)
    counts_outer = counts @ counts.T                       # (vocab_size, vocab_size)
    vocab_attn = indicator @ attention_matrix @ indicator.T  # (vocab_size, vocab_size)

    with np.errstate(invalid="ignore", divide="ignore"):
        vocab_attn = np.where(counts_outer > 0, vocab_attn / counts_outer, 0.0)

    return vocab_attn.astype(np.float32)


def sigmoid_contrast(x):
    """
    Apply contrast enhancement: f(x) = -2 / (1 + e^(4x)) + 1
    Maps [0, 1] -> [0, ~0.96], strongly boosts low values, never clips.
    """
    return -2.0 / (1.0 + np.exp(4.0 * x)) + 1.0


def attention_to_image(vocab_attn_matrix, config=None):
    """Convert a (256, 256) attention matrix to a grayscale PIL Image."""
    values = vocab_attn_matrix.copy()

    if config is not None:
        image_cfg = config.get("image", {})
        if image_cfg.get("contrast_enhancement", False):
            mode = image_cfg.get("contrast_mode", "sigmoid")
            if mode == "sigmoid":
                values = sigmoid_contrast(values)

    img_array = (values * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode="L")


def process_batch(bert_layer, batch_chunks, config):
    """
    Run a forward pass on a batch of chunks and return vocab-level attention matrices
    for each chunk and each (layer, head) pair.
    Returns list of dicts [{(layer_idx, head_idx): vocab_attn}, ...]
    """
    import tensorflow as tf

    pad_token_id = config["transformer"]["pad_token_id"]
    vocab_size = 2 ** config["transformer"]["token_bits"]

    input_ids = tf.constant(batch_chunks, dtype=tf.int32)
    attention_mask = tf.cast(tf.not_equal(input_ids, pad_token_id), dtype=tf.int32)

    outputs = bert_layer(input_ids, attention_mask=attention_mask, output_attentions=True)

    batch_results = []
    batch_size = len(batch_chunks)

    for b in range(batch_size):
        # Pass full token sequence — PAD tokens (value=256) are excluded
        # naturally since indicator only covers vocab values 0-255
        tokens = np.array(batch_chunks[b])

        chunk_attentions = {}
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            attn_np = layer_attn[b].numpy()
            for head_idx in range(attn_np.shape[0]):
                head_matrix = attn_np[head_idx]
                vocab_attn = aggregate_attention_to_vocab(head_matrix, tokens, vocab_size)
                chunk_attentions[(layer_idx, head_idx)] = vocab_attn

        batch_results.append(chunk_attentions)

    return batch_results


def generate_images_for_file(model, file_chunks, config, batch_size=4, file_label="", file_idx=0, total_files=0):
    """
    Generate 48 images (6 layers x 8 heads) for a single file.
    Uses batched inference and incremental RMS accumulation.
    Returns a dict {(layer_idx, head_idx): PIL Image}.
    """
    from transformers import TFBertModel
    bert_layer = next(l for l in model.layers if isinstance(l, TFBertModel))

    n_chunks = len(file_chunks)
    # Incremental RMS: accumulate sum of squares and count
    acc_sq = {}
    n_processed = 0

    for batch_start in range(0, n_chunks, batch_size):
        batch = file_chunks[batch_start:batch_start + batch_size]
        batch_results = process_batch(bert_layer, batch, config)

        for chunk_attentions in batch_results:
            for key, vocab_attn in chunk_attentions.items():
                if key not in acc_sq:
                    acc_sq[key] = np.zeros_like(vocab_attn)
                acc_sq[key] += vocab_attn ** 2
            n_processed += 1

        if batch_start % (batch_size * 10) == 0 or batch_start + batch_size >= n_chunks:
            print(f"    [{file_label} {file_idx+1}/{total_files}] chunk {min(batch_start + batch_size, n_chunks)}/{n_chunks} ...", flush=True)

    # Finalize RMS
    images = {}
    for (layer_idx, head_idx), sq_sum in acc_sq.items():
        rms_matrix = np.sqrt(sq_sum / n_processed)
        images[(layer_idx, head_idx)] = attention_to_image(rms_matrix, config=config)

    return images


def generate_and_save_images(model, files_chunks, labels, output_dir, config, batch_size=4, label_names=None):
    """
    Generate 48 attention images per file and save them organized by label.
    files_chunks: list of lists, each inner list contains the chunks for one file.
    """
    if label_names is None:
        label_names = {0: "benign", 1: "malware"}

    output_dir = Path(output_dir)
    for label_name in label_names.values():
        (output_dir / label_name).mkdir(parents=True, exist_ok=True)

    total = len(files_chunks)
    for file_idx, (file_chunks, label) in enumerate(zip(files_chunks, labels)):
        label_name = label_names[int(label)]
        print(f"\n[{file_idx+1}/{total}] {label_name} — {len(file_chunks)} chunks", flush=True)

        images = generate_images_for_file(
            model, file_chunks, config,
            batch_size=batch_size,
            file_label=label_name, file_idx=file_idx, total_files=total
        )

        for (layer_idx, head_idx), img in images.items():
            filename = f"file{file_idx:05d}_layer{layer_idx}_head{head_idx}.png"
            img.save(output_dir / label_name / filename)

        print(f"    -> {len(images)} images saved to {output_dir / label_name}/", flush=True)

    print("\nImage generation complete.")
