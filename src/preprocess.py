import re
import pandas as pd
from transformers import DistilBertTokenizer

# ── Constants ──────────────────────────────────────────────
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128

# Integer encoding for labels
LABEL2ID = {
    'normal':     0,
    'offensive':  1,
    'hatespeech': 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()} # Flips the previous dictionary so key: value becomes value: key

# All domain-specific special tokens identified during EDA
DOMAIN_SPECIAL_TOKENS = [
    '<user>', '<number>', '<percent>', '<censored>', '<date>',
    '<money>', '<time>', '<happy>', '<sad>', '<wink>', '<laugh>',
    '<url>', '<phone>', '<email>', '<annoyed>', '<will>',
    '<tong>', '<surprise>', '<kiss>', '<angel>'
]

def load_tokenizer(model_path: str = MODEL_NAME) -> DistilBertTokenizer:
    """
    Load the DistilBERT tokenizer and register domain-specific
    special tokens found during EDA. This ensures tokens like
    <user> and <censored> are treated as atomic units rather
    than being split into subword pieces.

    Args:
        model_path: HuggingFace model name or local directory path.
                    Defaults to MODEL_NAME (used during training).
                    Pass a local path to load a saved tokeniser that
                    already has special tokens registered.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    if model_path == MODEL_NAME:
        tokenizer.add_tokens(DOMAIN_SPECIAL_TOKENS)
        
    return tokenizer

def normalise_text(text: str) -> str:
    """
    Lightweight normalisation for HateXplain post tokens.

    Deliberately minimal — DistilBERT's tokenizer handles most
    normalisation internally (lowercasing, punctuation). We only
    fix structural issues specific to this dataset.
    """
    if not isinstance(text, str):
        return ""

    # Normalise whitespace — multiple spaces, tabs, newlines → single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Lowercase — distilbert-base-uncased expects lowercase input
    text = text.lower()

    return text

def tokenise(texts: list[str], tokenizer: DistilBertTokenizer) -> dict:
    """
    Convert a list of text strings into model-ready tensors.

    Returns a dict with:
        input_ids      — token integer IDs
        attention_mask — 1 for real tokens, 0 for padding
    """
    return tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=False   # not needed for classification
    )

def encode_labels(labels: list[str]) -> list[int]:
    """
    Convert string labels to integer IDs for model training.
    Raises a clear error if an unknown label is encountered
    rather than silently failing.
    """
    encoded = []
    for label in labels:
        if label not in LABEL2ID:
            raise ValueError(
                f"Unknown label '{label}'. "
                f"Expected one of: {list(LABEL2ID.keys())}"
            )
        encoded.append(LABEL2ID[label])
    return encoded


def decode_labels(ids: list[int]) -> list[str]:
    """
    Convert integer IDs back to human-readable label strings.
    Used during inference to return meaningful predictions.
    """
    return [ID2LABEL[i] for i in ids]

def preprocess(
    df: pd.DataFrame,
    tokenizer: DistilBertTokenizer,
    has_labels: bool = True
) -> dict:
    """
    Full preprocessing pipeline. Accepts a DataFrame and returns
    a dict of tensors ready for training or inference.

    Args:
        df:          DataFrame with 'post_tokens' column,
                     and optionally a 'label' column
        tokenizer:   Loaded tokenizer from load_tokenizer()
        has_labels:  Set False during inference when no labels exist

    Returns:
        dict with 'input_ids', 'attention_mask',
        and optionally 'labels'
    """
    # Step 1 — normalise text
    texts = df['post_tokens'].apply(normalise_text).tolist()

    # Step 2 — tokenise
    encodings = tokenise(texts, tokenizer)

    # Step 3 — encode labels if present
    result = {
        'input_ids':      encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
    }

    if has_labels:
        result['labels'] = encode_labels(df['label'].tolist())

    return result

if __name__ == '__main__':
    # Quick smoke test — run with: python src/preprocess.py
    df_test = pd.DataFrame({
        'post_tokens': [
            '<user> you are a <censored> go back to your country',
            'i love this weather today <happy>',
            'those <user> people are all the same <number> percent of them'
        ],
        'label': ['hatespeech', 'normal', 'offensive']
    })

    tokenizer = load_tokenizer()
    result = preprocess(df_test, tokenizer)

    print("input_ids shape:      ", result['input_ids'].shape)
    print("attention_mask shape: ", result['attention_mask'].shape)
    print("labels:               ", result['labels'])
    print("decoded labels:       ", decode_labels(result['labels']))
    print("\nSmoke test passed ✓")