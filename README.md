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
| Method B (Semantic Entropy) | 0.6041 | 0.2310 | 0.2121 |
| **Method C (MLP Combined)** | **0.7644** | **0.5756** | **0.4251** |

Method C improves AUROC by **+10.1pp** over the best individual method (A), and F1 by **+24.2pp** — confirming that combining linguistic, entropy, and probability signals captures failure modes that no single method handles alone.

### Per-Dataset AUROC

| Method | TriviaQA (n=806) | TruthfulQA (n=76) |
|--------|------------------|-------------------|
| Method A | 0.6064 | 0.5439 |
| Method B | 0.6346 | 0.5630 |

Method B outperforms Method A on both datasets individually, even though Method A leads on the aggregate — driven by TriviaQA's dominance (91% of data).

### Error Complementarity (Test Set)

| | Count |
|--|-------|
| Both A & B correct | 547 |
| A wrong, B correct | 82 |
| B wrong, A correct | 67 |
| Both wrong | 186 |

**Method C recovers 83 of the 186 cases** where both A and B fail — demonstrating that the combined model learns interaction effects neither component can capture alone. Method C total accuracy: **70.1% (618/882)**.

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
├── llm-hallucination-detector.ipynb  # Full pipeline notebook
├── metrics_summary.csv               # Aggregate results table
├── test_results.csv                  # Per-sample predictions
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
