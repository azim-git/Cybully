This project's aim is to build a multi-label text classifier on a real-world dataset.

This project will show: 
- Full preprocessing pipeline: tokenization, cleaning, handling class imbalance
- Fine-tune a small transformer (DistilBERT or RoBERTa) via HuggingFace Trainer
- Evaluation with precision/recall/F1, confusion matrix, error analysis
- Export and version your model with MLflow or Weights & Biases

Kaggle: https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification/data?select=hateXplain.csv
Data is stored in data/raw/ folder and processed into data/raw/

Commands:
python src/train.py