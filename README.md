# Training Efficiency in Fine-Tuning Transformers

> **Course:** Deep Learning and Applications — Part 2  
> **Instructor:** Prof. Diego Angelo Gaetano Reforgiato Recupero  
> **Program:** M.Sc. in Applied Artificial Intelligence — A.Y. 2024/2025  
> **Author:** Marco Sau

---

## What This Project Does

This project is a **systematic study of computational efficiency in transformer fine-tuning**. It replicates the fine-tuning pipeline from Chapter 3 of the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) and extends it with a controlled multi-model, multi-strategy experiment.

We answer the question:

> *When do model architecture choices and data preprocessing strategies offer meaningful computational benefits during fine-tuning?*

---

## Research Hypothesis

Before running the experiments, we hypothesised:

> *Dynamic padding (which pads each batch only to its longest sequence) will save significant training time on datasets with variable-length text (IMDb), but provide minimal benefit on datasets with short, uniform text (Emotion).*

**What we actually found:** The benefit of dynamic padding is not determined by *length variability* but by the **padding efficiency ratio** `η = mean_length / max_length`. The lower `η` is, the more padding is wasted under static padding, and the more dynamic padding helps. See the report for the full analysis.

---

## Experimental Setup

| Dimension | Values |
|---|---|
| Models | BERT-base-uncased, DistilBERT-base-uncased, RoBERTa-base |
| Padding strategies | Static, Dynamic |
| Datasets | IMDb (binary, long text), Emotion (6-class, short text) |
| Seeds | 42, 123, 456 |
| **Total runs** | **3 × 2 × 2 × 3 = 36** |

---

## Key Results

### IMDb (max_length = 512, η ≈ 0.54)

| Model | Strategy | Accuracy | Time (min) |
|---|---|---|---|
| BERT-base | Static | 93.97% | 32.82 |
| BERT-base | Dynamic | 93.87% | 31.57 (-4%) |
| DistilBERT | Static | 93.07% | 16.80 |
| DistilBERT | Dynamic | 93.00% | 16.30 (-3%) |
| RoBERTa-base | Static | **95.35%** | 33.10 |
| RoBERTa-base | Dynamic | **95.30%** | 31.48 (-5%) |

### Emotion (max_length = 128, η ≈ 0.19)

| Model | Strategy | Accuracy | Time (min) |
|---|---|---|---|
| BERT-base | Static | 92.70% | 8.53 |
| BERT-base | Dynamic | 92.83% | 7.63 **(-11%)** |
| DistilBERT | Static | 92.62% | 4.81 |
| DistilBERT | Dynamic | 92.75% | 4.34 **(-10%)** |
| RoBERTa-base | Static | 92.92% | 8.94 |
| RoBERTa-base | Dynamic | 92.90% | 7.97 **(-11%)** |

**Key takeaways:**
- DistilBERT is ~1.94× faster than BERT with only -0.89% accuracy loss
- RoBERTa matches BERT in training time but is consistently +1.41% more accurate
- Dynamic padding saves ~4% time on IMDb (η≈0.54) vs ~11% on Emotion (η≈0.19)
- Accuracy is never meaningfully affected by the choice of padding strategy

---

## Project Structure

```
Project/
│
├── configs/                        # Experiment configuration files
│   ├── base_config.yaml            # Baseline config (1 epoch, batch=32, BERT+DistilBERT)
│   └── extended_imdb_config.yaml   # Main experiment config (3 epochs, batch=8, all 3 models)
│
├── notebooks/
│   └── colab_notebook_extended.ipynb   # Main Colab notebook (run this!)
│
├── src/                            # Python source modules
│   ├── __init__.py
│   ├── config.py           # YAML config loading and dataclass definitions
│   ├── data_processing.py  # Dataset loading, tokenization, padding collators
│   ├── model_utils.py      # Model loading from Hugging Face Hub
│   ├── training.py         # Trainer setup, TrainingArguments, training loop
│   ├── evaluation.py       # Metrics, inference latency, GPU memory measurement
│   ├── experiment_runner.py # High-level orchestration: runs all seeds & strategies
│   └── visualization.py    # Plotting utilities (not used in Colab run)
│
├── results/
│   └── tables/
│       ├── padding_comparison_imdb.csv     # IMDb results (18 rows)
│       └── padding_comparison_emotion.csv  # Emotion results (18 rows)
│
├── run_experiments.py  # CLI entry point (used for local execution)
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── QUICKSTART.md       # Fast reproduction guide
├── COLAB_SETUP.md      # Detailed step-by-step Colab guide for beginners
└── PROPOSAL_ALIGNMENT.md  # Checklist verifying implementation vs. proposal
```

---

## Source Module Descriptions

### `src/config.py`
Loads YAML configuration files into Python dataclasses. Defines `TrainingConfig`, `DatasetConfig`, and `ExperimentConfig`. All hyperparameters (learning rate, batch size, seeds, max_length, etc.) flow from the YAML files through this module.

### `src/data_processing.py`
Contains `load_and_preprocess_imdb()` and `load_and_preprocess_emotion()`. Each function:
1. Downloads the dataset from Hugging Face Datasets
2. Splits train into train/validation (90/10)
3. Tokenizes using the model's tokenizer (truncation to `max_length`)
4. Renames the `label` column to `labels` (required by Trainer)
5. Returns a `DatasetDict` with `train`, `validation`, and `test` splits

Also defines `StaticPaddingCollator` (pads to `max_length`) and wraps `DataCollatorWithPadding` for dynamic padding.

### `src/model_utils.py`
Loads pre-trained models via `AutoModelForSequenceClassification.from_pretrained()`. Handles the classification head initialization for the correct number of labels.

### `src/training.py`
Wraps the Hugging Face `Trainer` API. Defines `TrainingArguments` from config values, trains the model, and returns training time and GPU memory usage. Key settings: FP16 mixed precision, per-epoch evaluation, best-model checkpoint restoration.

### `src/evaluation.py`
Provides `evaluate_model_comprehensive()` (runs inference on the full test set, records latency and GPU memory) and `compute_metrics()` (accuracy, F1-macro, F1-micro) for use with the Trainer's `compute_metrics` callback.

### `src/experiment_runner.py`
The top-level orchestration function `run_padding_comparison()`. For each `(model, padding_strategy, seed)` combination: sets random seeds, loads data, loads model, trains, evaluates, records all metrics to a DataFrame, and appends results to a CSV file.

---

## Hardware Requirements

All experiments were run on **Google Colab Pro** (paid tier) using an **NVIDIA T4 GPU (16 GB VRAM)**. The free Colab tier does not provide sufficient session duration or GPU memory for the full 36-run experiment (~18 hours of GPU time total).

See `COLAB_SETUP.md` for full setup instructions.
