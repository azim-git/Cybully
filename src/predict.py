import torch
from transformers import DistilBertForSequenceClassification
from preprocess import load_tokenizer, normalise_text, LABEL2ID

ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# in-memory cache: model_name → (model, tokeniser)
_cache: dict = {}


def load_model(model_dir: str):
    """
    Load model and tokeniser from disk, caching after first load.

    Args:
        model_dir: path to a saved model directory e.g. 'models/v2_class_weights'

    Returns:
        (model, tokeniser) tuple
    """
    if model_dir not in _cache:
        tokeniser = load_tokenizer(model_dir)
        model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        model.to(DEVICE)
        model.eval()
        _cache[model_dir] = (model, tokeniser)

    return _cache[model_dir]


def predict(text: str, model_dir: str) -> dict:
    """
    Run inference on a single text string.

    Args:
        text:      raw user-submitted post
        model_dir: path to saved model directory

    Returns:
        {
            "label":      "normal" | "offensive" | "hatespeech",
            "confidence": float,
            "scores": {
                "normal":     float,
                "offensive":  float,
                "hatespeech": float
            }
        }
    """
    model, tokeniser = load_model(model_dir)

    # normalise text the same way training data was normalised
    cleaned = normalise_text(text)

    # tokenise — same settings as training
    encoding = tokeniser(
        cleaned,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"     # return PyTorch tensors directly
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

    predicted_idx = probs.index(max(probs))

    return {
        "label":      ID2LABEL[predicted_idx],
        "confidence": round(max(probs), 4),
        "scores": {
            ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)
        }
    }


if __name__ == "__main__":
    # quick smoke test
    test_text = "I hate all people from that country, they should leave"
    result    = predict(test_text, "models/v2_class_weights")
    print(result)