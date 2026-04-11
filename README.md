# 🔍 LLM Hallucination Detector

Detecting when a large language model is guessing rather than answering from reliable knowledge — using linguistic signals, semantic consistency, and token-level probabilities.

**CSCI 544 — Natural Language Processing | University of Southern California**  
*Dhruvi Jodhawat, Harsh Mangukiya, Keshav Mannem, Manvika Satish, Shuban Sridhar*

---

## Overview

Large language models frequently produce fluent, confident responses that are factually wrong. This project builds a system that assigns a **guessing score** to any LLM response — indicating how likely it is to be incorrect — by combining three complementary uncertainty signals.

---

## Methods

### Method A — Linguistic Feature Classifier
Extracts 9 surface-level features per response (hedge word frequency, modal verb count, answer length, first-person uncertainty markers, punctuation cues) and trains an **XGBoost classifier** on them. Fast and interpretable, but blind to overconfident hallucinations.

### Method B — Consistency-Based Uncertainty
Generates **K=5 temperature-sampled responses** per question, embeds them with **Sentence-BERT** (`all-MiniLM-L6-v2`), clusters with agglomerative clustering, and computes **Shannon entropy** over the cluster distribution. Captures uncertainty invisible to logit-based approaches, at 5× inference cost.

### Method C — Combined Signal Model (MLP)
Concatenates features from Methods A and B with **token-level log-probabilities** from greedy inference into a unified representation, then trains a **4-layer MLP** with `BCEWithLogitsLoss` and class weighting to address the 69/31 grounded/guessing imbalance. Threshold is tuned on the dev set.

---

## Dataset

| Dataset | Samples | Guessing Rate | Purpose |
|---------|---------|---------------|---------|
| TriviaQA | 8,000 | 25.7% | Factual recall |
| TruthfulQA | 817 | 87.1% | Hallucination-prone |
| **Total** | **8,817** | **31.4%** | |

Split: 80 / 10 / 10 (stratified) → 7,053 train / 882 dev / 882 test

---

## Results

### Overall Test Performance

| Method | AUROC | F1 | Spearman |
|--------|-------|----|----------|
| Majority Baseline | 0.5000 | 0.0000 | — |
| Method A (XGBoost) | 0.6632 | 0.3333 | 0.2625 |
| Method B (Semantic Entropy) | 0.6018 | 0.2624 | 0.2049 |
| Method C (MLP Combined) | — | — | — |

### Per-Dataset AUROC

| Method | TriviaQA | TruthfulQA |
|--------|----------|------------|
| Method A | 0.6064 | 0.5439 |
| Method B | 0.6297 | 0.6269 |

Method B outperforms Method A on both datasets individually, even though Method A leads on the aggregate — driven by TriviaQA's dominance (91% of data).

### Error Complementarity
- **87 cases** where A failed but B succeeded
- **72 cases** where B failed but A succeeded
- **181 cases** where both failed → primary target for Method C

---

## Model & Infrastructure

- **LLM:** Mistral-7B-Instruct-v0.3 (fp16, HuggingFace Transformers)
- **Hardware:** NVIDIA RTX 5090 (34.2 GB VRAM)
- **Greedy inference:** ~14 minutes over 8,817 questions
- **Sampled inference (K=5):** ~62 minutes

---

## Project Structure

```
llm-hallucination-detector/
├── llm-hallucination-detector.ipynb   # Full pipeline notebook
├── metrics_summary.csv              # Aggregate results table
├── test_results.csv                 # Per-sample predictions
└── figures/
    ├── 01_label_distribution.png
    ├── 02_entropy_distribution.png
    ├── 03_feature_importance.png
    ├── 04_roc_curves.png
    ├── 05_metrics_comparison.png
    └── 06_error_complementarity.png
```

---

## Reproducing Results

**First run (no cache):**
```bash
# Install dependencies
pip install transformers accelerate datasets sentence-transformers xgboost scikit-learn scipy bitsandbytes

# Run the notebook top to bottom
jupyter notebook llm-hallucination-detector.ipynb
```

**Subsequent runs (with cache):** Inference outputs are cached automatically. To skip the expensive steps, uncomment the resume block in cell 16 after the first run — this restores `token_logprobs` and `labeled_data` without re-querying Mistral.

Cached files (not in repo, generated on first run):
- `model_outputs.pkl` — greedy inference outputs
- `token_logprobs.pkl` — token-level log-probabilities
- `sampled_answers.json` — K=5 sampled responses
- `mlp_best.pt` — best Method C checkpoint

---

## References

1. Nguyen et al. (2025). Beyond Semantic Entropy. ACL Findings.
2. Azaria & Mitchell (2023). The Internal State of an LLM Knows When It's Lying. arXiv.
3. Tan et al. (2024). Too Consistent to Detect. EMNLP.
4. Kossen et al. (2024). Semantic Entropy Probes. arXiv.
5. Manakul et al. (2023). SelfCheckGPT. EMNLP.
