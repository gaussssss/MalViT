import yaml
import numpy as np
import tensorflow as tf
from transformers import BertConfig, TFBertModel


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_vocab_size(token_bits, pad_token_id):
    return max(2 ** token_bits, pad_token_id) + 1


def build_attention_mask(input_ids, pad_token_id):
    """Return attention mask: 1 for real tokens, 0 for PAD tokens."""
    return tf.cast(tf.not_equal(input_ids, pad_token_id), dtype=tf.int32)


def build_model(config):
    """Build and compile the BERT-based binary classifier."""
    transformer_cfg = config["transformer"]

    vocab_size = compute_vocab_size(
        transformer_cfg["token_bits"],
        transformer_cfg["pad_token_id"]
    )

    bert_config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=transformer_cfg["max_seq_len"],
        num_hidden_layers=transformer_cfg["num_layers"],
        num_attention_heads=transformer_cfg["num_heads"],
        hidden_size=transformer_cfg["hidden_size"],
        intermediate_size=transformer_cfg["hidden_size"] * 4,
        output_attentions=True,
    )

    pad_token_id = transformer_cfg["pad_token_id"]
    max_seq_len = transformer_cfg["max_seq_len"]

    input_ids = tf.keras.Input(shape=(max_seq_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(tf.not_equal(x, pad_token_id), dtype=tf.int32),
        name="attention_mask"
    )(input_ids)

    bert = TFBertModel(bert_config, name="bert")
    outputs = bert(input_ids, attention_mask=attention_mask)

    sequence_output = outputs.last_hidden_state
    pooled = tf.keras.layers.GlobalAveragePooling1D(name="mean_pooling")(sequence_output)

    logits = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(pooled)

    model = tf.keras.Model(inputs=input_ids, outputs=logits)
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
    model = build_model(config)
    model.summary()
