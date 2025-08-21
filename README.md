# NLP Assignment – Sentiment Classification (IMDb)

This project evaluates a pretrained Transformer on the IMDb movie-reviews dataset and saves metrics + artifacts for submission.

## Contents
- `main.py` – loads IMDb, builds a balanced test subset, runs inference with a pretrained model, saves results.
- `prompts.py` – (intentionally empty for this assignment).
- `requirements.txt` – Python dependencies.
- `outputs/`
  - `metrics.json` – accuracy, precision, recall, F1.
  - `confusion_matrix.png` – 2×2 confusion matrix image.
  - `sample_predictions.csv` – text snippets + true/pred labels + positive probability.

## Dataset
- **IMDb** (50k reviews, binary labels). Fetched automatically via 🤗 `datasets`—no manual download needed.

## Model
- `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face). Used as a lightweight, strong sentiment baseline.

---

## How to run

### 1) Create & activate a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
