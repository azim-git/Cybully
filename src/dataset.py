import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocess import (
    load_tokenizer,
    preprocess,
    LABEL2ID
)


class HateXplainDataset(Dataset):
    """
    PyTorch Dataset for the HateXplain cyberbullying corpus.

    Wraps preprocessed encodings and labels into a format
    PyTorch's DataLoader can iterate over during training.
    """

    def __init__(self, encodings: dict, labels: list[int]):
        """
        Args:
            encodings: dict with 'input_ids' and 'attention_mask' tensors
            labels:    list of integer encoded labels
        """
        self.input_ids      = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels':         self.labels[idx]
        }
    

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split clean_posts into train / validation / test sets.

    Stratified on label to preserve class distribution across splits.

    Args:
        df:           clean_posts DataFrame
        test_size:    proportion for test set  (default 15%)
        val_size:     proportion for val set   (default 15%)
        random_state: seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )

    # Then split train into train + validation
    # val_size is relative to the remaining train_val data
    relative_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )

    return train_df, val_df, test_df

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Preprocess each split and wrap in DataLoaders.

    Args:
        train_df, val_df, test_df: split DataFrames
        tokenizer:                 loaded tokenizer
        batch_size:                samples per training batch

    Returns:
        train_loader, val_loader, test_loader
    """
    def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        encodings = preprocess(df, tokenizer, has_labels=True)
        labels    = encodings.pop('labels')
        dataset   = HateXplainDataset(encodings, labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0        # set to 2-4 on Linux for faster loading
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df,   shuffle=False)
    test_loader  = make_loader(test_df,  shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import sys
    import os

    # Get the directory where dataset.py lives (src/)
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))

    # Get the project root (one level up from src/)
    PROJECT_ROOT = os.path.dirname(SRC_DIR)

    # Build absolute paths from the project root
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    # Load clean data
    df = pd.read_csv(os.path.join(DATA_DIR, 'clean_posts.csv'))

    # Parse target_groups back from string to list
    # (pd.read_csv stringifies lists on save)
    print(f"Total posts: {len(df):,}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # Split
    train_df, val_df, test_df = split_dataset(df)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"\nTrain label distribution:")
    print(train_df['label'].value_counts(normalize=True).mul(100).round(2))
    print(f"\nTest label distribution:")
    print(test_df['label'].value_counts(normalize=True).mul(100).round(2))

    # Build dataloaders
    tokenizer = load_tokenizer()
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=32
    )

    # Inspect one batch
    batch = next(iter(train_loader))
    print(f"\nOne batch:")
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape:         {batch['labels'].shape}")
    print(f"  labels sample:        {batch['labels'][:8]}")

    print("\nSmoke test passed ✓")