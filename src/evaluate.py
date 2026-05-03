import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from preprocess import load_tokenizer, ID2LABEL
from dataset import split_dataset, build_dataloaders


def run_inference(
    model: DistilBertForSequenceClassification,
    loader: DataLoader,
    device: torch.device
) -> tuple[list[int], list[int], list[list[float]]]:
    """
    Run model over a DataLoader and collect predictions.

    Returns:
        true_labels:  ground truth integer labels
        pred_labels:  predicted integer labels
        confidences:  softmax probability scores [n_samples x 3]
    """
    model.eval()

    true_labels  = []
    pred_labels  = []
    confidences  = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']           # keep on CPU for collection

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu()  # move back to CPU
            preds   = torch.argmax(probs, dim=-1)

            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.tolist())
            confidences.extend(probs.tolist())

    return true_labels, pred_labels, confidences


def print_metrics(true_labels: list[int], pred_labels: list[int]) -> None:
    """Print accuracy and per-class F1 scores."""
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            pred_labels,
            target_names=[ID2LABEL[i] for i in range(3)]
        )
    )


def plot_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    output_path: str = "models/confusion_matrix.png"
) -> None:
    """Plot and save a normalised confusion matrix."""
    labels     = [ID2LABEL[i] for i in range(3)]
    cm         = confusion_matrix(true_labels, pred_labels)
    cm_norm    = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalise

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (row-normalised)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to '{output_path}'")


def evaluate(
    data_path:  str = "data/processed/clean_posts.csv",
    model_dir:  str = "models/v2_class_weights",
) -> None:
    """
    Load saved checkpoint, run on test set, print metrics, save confusion matrix.
    """
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model + tokeniser ────────────────────────────────────────────────
    tokeniser = load_tokenizer(model_dir)   # load from saved dir — has special tokens
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    print(f"Model loaded from '{model_dir}'")

    # ── Test data ─────────────────────────────────────────────────────────────
    df                          = pd.read_csv(data_path)
    train_df, val_df, test_df   = split_dataset(df)
    _, _, test_loader           = build_dataloaders(train_df, val_df, test_df, tokeniser)
    print(f"Test set: {len(test_df):,} posts")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning inference on test set...")
    true_labels, pred_labels, confidences = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print_metrics(true_labels, pred_labels)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(true_labels, pred_labels)


if __name__ == "__main__":
    evaluate()