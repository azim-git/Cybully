import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader

from preprocess import load_tokenizer, preprocess, ID2LABEL
from dataset import HateXplainDataset

def build_ambiguous_loader(
    df: pd.DataFrame,
    tokeniser,
    batch_size: int = 32
) -> DataLoader:
    """
    Build a DataLoader for ambiguous posts.
    These have no labels — we pass has_labels=False.
    """
    encodings = preprocess(df, tokeniser, has_labels=False)
    dataset   = HateXplainDataset(
        encodings,
        labels=[0] * len(df)    # dummy labels — never used
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_ambiguous_inference(
    model: DistilBertForSequenceClassification,
    loader: DataLoader,
    device: torch.device
) -> list[list[float]]:
    """Run inference and return softmax confidence scores."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=-1).cpu()
            all_probs.extend(probs.tolist())

    return all_probs


def analyse(
    ambiguous_path: str = "data/processed/ambiguous_posts.csv",
    model_dir:      str = "models/"
) -> None:

    # ── Setup ─────────────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokeniser = load_tokenizer(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    df     = pd.read_csv(ambiguous_path)
    loader = build_ambiguous_loader(df, tokeniser)
    print(f"Ambiguous posts: {len(df):,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    probs = run_ambiguous_inference(model, loader, device)

    # attach results back to the dataframe
    df['p_normal']     = [p[0] for p in probs]
    df['p_offensive']  = [p[1] for p in probs]
    df['p_hatespeech'] = [p[2] for p in probs]
    df['predicted']    = df[['p_normal','p_offensive','p_hatespeech']].idxmax(axis=1)
    df['predicted']    = df['predicted'].str.replace('p_', '')
    df['confidence']   = df[['p_normal','p_offensive','p_hatespeech']].max(axis=1)

    # ── 1. Predicted label distribution ───────────────────────────────────────
    print("\n── Predicted label distribution ──────────────────────────────")
    dist = df['predicted'].value_counts()
    print(dist.to_string())
    print(f"\nAs percentages:")
    print((dist / len(df) * 100).round(1).to_string())

    # ── 2. Confidence distribution ────────────────────────────────────────────
    print("\n── Confidence score summary ──────────────────────────────────")
    print(df['confidence'].describe().round(3).to_string())

    low_conf  = (df['confidence'] < 0.50).sum()
    high_conf = (df['confidence'] > 0.80).sum()
    print(f"\nLow confidence  (<0.50): {low_conf:,} posts ({low_conf/len(df)*100:.1f}%)")
    print(f"High confidence (>0.80): {high_conf:,} posts ({high_conf/len(df)*100:.1f}%)")

    # ── 3. Most confidently predicted examples ────────────────────────────────
    print("\n── 5 most confident predictions ──────────────────────────────")
    top5 = df.nlargest(5, 'confidence')[['post_tokens','predicted','confidence']]
    for _, row in top5.iterrows():
        print(f"\n  [{row['predicted'].upper()} | {row['confidence']:.3f}]")
        print(f"  {str(row['post_tokens'])[:120]}")

    # ── 4. Most uncertain examples ────────────────────────────────────────────
    print("\n── 5 most uncertain predictions ──────────────────────────────")
    bot5 = df.nsmallest(5, 'confidence')[['post_tokens','predicted','confidence',
                                          'p_normal','p_offensive','p_hatespeech']]
    for _, row in bot5.iterrows():
        print(f"\n  [{row['predicted'].upper()} | conf {row['confidence']:.3f}]")
        print(f"  normal {row['p_normal']:.3f} | offensive {row['p_offensive']:.3f} | hatespeech {row['p_hatespeech']:.3f}")
        print(f"  {str(row['post_tokens'])[:120]}")

    # ── 5. Confidence histogram ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # overall confidence distribution
    axes[0].hist(df['confidence'], bins=30, edgecolor='black', color='steelblue')
    axes[0].set_title("Confidence Distribution (Ambiguous Posts)")
    axes[0].set_xlabel("Max softmax probability")
    axes[0].set_ylabel("Count")
    axes[0].axvline(0.5, color='red', linestyle='--', label='0.5 threshold')
    axes[0].legend()

    # per-class average confidence
    avg_conf = df[['p_normal','p_offensive','p_hatespeech']].mean()
    axes[1].bar(
        ['normal','offensive','hatespeech'],
        avg_conf.values,
        color=['steelblue','orange','firebrick'],
        edgecolor='black'
    )
    axes[1].set_title("Average Class Probability (Ambiguous Posts)")
    axes[1].set_ylabel("Mean softmax probability")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("models/ambiguous_analysis.png", dpi=150)
    plt.close()
    print("\nPlot saved to 'models/ambiguous_analysis.png'")


if __name__ == "__main__":
    analyse()