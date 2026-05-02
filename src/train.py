import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
import pandas as pd


def train_one_epoch(
    model: DistilBertForSequenceClassification,
    train_loader: DataLoader,
    optimiser: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training set.

    Returns:
        Average training loss across all batches.
    """
    model.train()   # activates dropout and batch norm — important

    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=False):

        # move batch tensors to the same device as the model
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimiser.zero_grad()   # clear gradients from previous batch

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()                                              # compute gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()     # update weights
        scheduler.step()     # advance LR schedule

        total_loss += loss.item()   # .item() extracts a plain Python float from the tensor

    return total_loss / len(train_loader)   # average loss per batch

def evaluate(
    model: DistilBertForSequenceClassification,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Run one full pass over the validation set.

    Returns:
        Average validation loss across all batches.
    """
    model.eval()    # deactivates dropout — deterministic predictions

    total_loss = 0.0

    with torch.no_grad():   # no computation graph — saves memory and time
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(val_loader)

import os
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

def train(
    data_path: str  = "data/processed/clean_posts.csv",
    output_dir: str = "models/v2_class_weights",
    epochs: int     = 4,
    lr: float       = 2e-5,
    warmup_ratio: float = 0.1,
) -> None:
    """
    Full training pipeline: load data, fine-tune DistilBERT, save best checkpoint.

    Args:
        data_path:    path to clean_posts.csv
        output_dir:   directory to save the best model checkpoint
        epochs:       number of full passes over the training set
        lr:           peak learning rate for AdamW
        warmup_ratio: fraction of total steps used for LR warmup
    """

    # ── 1. Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 2. Data ───────────────────────────────────────────────────────────────
    # import here to avoid circular imports at module level
    from dataset import build_dataloaders, split_dataset
    from preprocess import load_tokenizer

    tokeniser                             = load_tokenizer()
    df                                    = pd.read_csv(data_path)
    train_df, val_df, test_df             = split_dataset(df)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, tokeniser)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3        # normal / offensive / hatespeech
    )
    model.resize_token_embeddings(len(tokeniser))   # account for 20 added special tokens
    model.to(device)

    # ── 4. Optimiser ──────────────────────────────────────────────────────────
    # weight_decay regularises all parameters except biases and LayerNorm weights
    # — applying decay to those can hurt performance, so we exclude them
    no_decay = ["bias", "LayerNorm.weight"]
    optimiser_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimiser = AdamW(optimiser_grouped_parameters, lr=lr)

    # ── 5. LR Scheduler ───────────────────────────────────────────────────────
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")    # any real loss will be lower than infinity

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimiser, scheduler, device)
        val_loss   = evaluate(model, val_loader, device)

        print(f"  train loss: {train_loss:.4f}  |  val loss: {val_loss:.4f}")

        # save checkpoint if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            tokeniser.save_pretrained(output_dir)
            print(f"  ✓ checkpoint saved (val loss: {best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f} — model saved to '{output_dir}'")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()