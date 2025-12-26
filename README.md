# FOMC Statement Reaction: Embedding + Model Comparison

## Research Question
Do FOMC post-meeting statements contain **any signal** about the **next trading day direction** of the S&P 500?

This is an embedding/model comparison project (not a trading system).

---

## Setup

### Create environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage
Run the full experiment end-to-end:
```bash
python main.py
```

### Expected output
- Prints progress checkpoints (load data → build embeddings → run models → save results)
- Writes result tables and artifacts to `results/`
- Optionally prompts for live predictions at the end (`y/n`)

---

## Project Structure
```
project_root/
├─ main.py                     # Main entry point (runs the full experiment)
├─ requirements.txt            # Python dependencies
├─ README.md
├─ src/
│  ├─ data_loader.py           # Load CSV + chronological split + X/y extraction
│  ├─ embedder_tfidf.py        # TF-IDF features
│  ├─ embedder_bert.py         # BERT embeddings
│  ├─ embedder_finbert.py      # FinBERT embeddings
│  ├─ embedder_gte.py          # GTE embeddings
│  ├─ embedding_cache.py       # Save/load cached embeddings
│  ├─ models.py                # Model wrappers (LR/RF/XGB + regressors)
│  ├─ evaluation.py            # Metrics + experiment loops + diagnostics
│  ├─ model_selection.py       # Select robust candidates ("Grand Champion")
│  ├─ inference.py             # Optional interactive inference
│  └─ data/
│     ├─ update_fomc_statements.py     # (Optional) download missing statements
│     └─ prepare_processed_data.py     # (Optional) rebuild processed CSV
├─ data/
│  ├─ interim/                 # One .txt per statement (YYYYMMDD.txt)
│  ├─ processed/
│  │  └─ fomc_statements.csv   # Final dataset used by main.py
│  └─ embeddings_cache/        # Created on first run (may be absent initially)
└─ results/                    # Generated outputs (CSVs + saved champion model)
```

---

## Results (what to look at)
Main metric is **ROC-AUC** (directional ranking). Each model family is evaluated under **three
fixed configurations** (conservative / moderate / aggressive) to assess robustness instead of
heavy tuning.

After running `main.py`, check:
- `results/experiment_results.csv`          — all runs (task × embedding × model × config)
- `results/diagnostic_breakdown.csv`        — mean AUC + robustness diagnostics + category
- `results/diagnostic_composition.csv`      — diagnostic composition table
- `results/grand_champion_model.pkl`        — saved final model for inference
- `results/grand_champion_config.json`      — metadata for inference
- `results/tfidf_vectorizer.pkl`            — only if TF-IDF is selected as champion

---

## Runtime notes (important)
Importing the embedder modules is fast. Runtime is dominated by:
1) **Loading transformer weights** (BERT/FinBERT/GTE). On the first run, if weights are not
   already cached locally, Hugging Face will download them once.
2) **Computing embeddings** for all statements (done in mini-batches on CPU).

Cached embeddings are stored in `data/embeddings_cache/` and reused on subsequent runs if
the dataset size is unchanged.

---

## Optional: Update data / rebuild processed dataset
If you want to refresh statements and rebuild the final CSV:

```bash
python src/data/update_fomc_statements.py
python src/data/prepare_processed_data.py
```

---

## Optional: Live inference
If you already ran `main.py` and saved the champion artifacts:

```bash
python -c "from src.inference import run_interactive_prediction; run_interactive_prediction()"
```
