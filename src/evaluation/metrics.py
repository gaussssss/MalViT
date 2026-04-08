import sys
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_thresholds(scores, config):
    """
    Apply decision thresholds from config to probability scores.
    Returns predicted labels (0=benign, 1=malware) and decisions.
    """
    block_threshold = config["thresholds"]["block"]
    quarantine_threshold = config["thresholds"]["quarantine"]

    decisions = []
    for score in scores:
        if score >= block_threshold:
            decisions.append("blocked")
        elif score >= quarantine_threshold:
            decisions.append("quarantine")
        else:
            decisions.append("allowed")

    y_pred = (scores >= quarantine_threshold).astype(int)
    return y_pred, decisions


def compute_metrics(y_true, y_pred, y_scores):
    """Compute and return all evaluation metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_scores),
    }


def print_metrics(metrics):
    print("\n--- Evaluation Results ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malware"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(y_true, y_scores, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"MalViT (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def evaluate(model, X_test, y_test, config, output_dir="results"):
    """Full evaluation pipeline: metrics + confusion matrix + ROC curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_scores = model.predict(X_test).flatten()
    y_pred, decisions = apply_thresholds(y_scores, config)

    metrics = compute_metrics(y_test, y_pred, y_scores)
    print_metrics(metrics)

    plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_scores, output_dir / "roc_curve.png")

    return metrics
