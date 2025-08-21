# main.py
import os, json, csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # strong pretrained sentiment model
MAX_LEN = 256
BATCH_SIZE = 32
N_SAMPLES = 1000            # total eval samples (balanced 50/50)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Load IMDb and build a BALANCED test subset
# -----------------------------
dataset = load_dataset("imdb")
test = dataset["test"].shuffle(seed=42)

pos = test.filter(lambda x: x["label"] == 1)
neg = test.filter(lambda x: x["label"] == 0)
n_each = min(N_SAMPLES // 2, len(pos), len(neg))

sample = concatenate_datasets([
    pos.select(range(n_each)),
    neg.select(range(n_each))
]).shuffle(seed=123)

texts = sample["text"]
labels = np.array(sample["label"])
print(f"Eval set size: {len(texts)} (pos={int(labels.sum())}, neg={len(labels)-int(labels.sum())})")

# -----------------------------
# 2) Load tokenizer + model (CPU or CUDA automatically)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -----------------------------
# 3) Predict in batches
# -----------------------------
all_preds, all_probs = [], []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        ).to(device)

        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.extend(probs[:, 1].tolist())  # prob of POS (label=1)
        all_preds.extend(preds.tolist())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# -----------------------------
# 4) Metrics
# -----------------------------
acc = accuracy_score(labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(labels, all_preds, average="binary", zero_division=0)

metrics = {
    "samples": int(len(labels)),
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4),
}
print("\nEvaluation:")
for k, v in metrics.items():
    print(f"{k}: {v}")

with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# -----------------------------
# 5) Confusion matrix image
# -----------------------------
cm = confusion_matrix(labels, all_preds, labels=[0, 1])  # 0=NEG, 1=POS
fig = plt.figure(figsize=(4, 4))
plt.imshow(cm)
plt.title("Confusion Matrix (IMDb balanced subset)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ["NEG", "POS"])
plt.yticks([0, 1], ["NEG", "POS"])

for (r, c), val in np.ndenumerate(cm):
    plt.text(c, r, int(val), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# -----------------------------
# 6) Sample predictions CSV
# -----------------------------
csv_path = os.path.join(OUT_DIR, "sample_predictions.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text_snippet", "true_label", "pred_label", "prob_positive"])
    for t, y, p, pr in zip(texts[:200], labels[:200], all_preds[:200], all_probs[:200]):
        snippet = " ".join(t.split())
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        w.writerow([snippet, int(y), int(p), round(float(pr), 4)])

print("\nSaved files in 'outputs/': metrics.json, confusion_matrix.png, sample_predictions.csv")
