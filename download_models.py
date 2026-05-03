from huggingface_hub import snapshot_download
import os

MODELS = {
    "models/baseline":         "AA_Hugger/cybully-baseline",
    "models/v2_class_weights": "AA_Hugger/cybully-v2",
    "models/v3_lower_lr":      "AA_Hugger/cybully-v3",
    "models/v4_more_epochs":   "AA_Hugger/cybully-v4",
}

for local_dir, repo_id in MODELS.items():
    if not os.path.exists(local_dir):
        print(f"Downloading {repo_id} → {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    else:
        print(f"Already exists: {local_dir}")