# Can LLMs Identify Primary Flight Incident Problems?

**Authors:** CSCI 49354 Project Team  
**Dataset:** NASA Aviation Safety Reporting System (ASRS) (2015–2025)

---

## Abstract

While aviation is acclaimed as one of the safest modes of travel, the extensive coordination between personnel, facilities, and machinery creates a high potential for failure. Analyzing safety incidents and identifying their leading causes is critical for future prevention. Previous research applied basic NLP strategies (TF-IDF) and older LLMs (GPT-3.5-Turbo) to predict primary problems from ASRS narrative reports. 

In this work, we:
1. **Leverage modern LLMs and methods:** Fine-tuned **ModernBERT**, **Gemma 4 (E2B-it)** with **LoRA**, **GPT OSS-120B**, and **RAG / multi-stage decision pipelines**.
2. **Utilize structured event tag data:** Feature-engineered event tags (150-D vectors) alongside free-form text narratives across **37,301 ASRS incidents (2015–2025)** to perform **18-class primary problem multiclass classification**.

**Key Finding:** Increased model and pipeline complexity **did not significantly outperform a simple TF-IDF + Logistic Regression baseline** (0.66 accuracy, 0.91 top-3 accuracy). Analysis reveals that "primary problem" assignment is inherently subjective and label definitions are ambiguous, establishing a performance ceiling rooted in dataset label noise rather than model capacity.

---

## System Architecture & Classification Methods

We explored two primary data representations: **Event Tags** (structured/human-labeled) and **Narratives** (unstructured free text).

```
                      ┌──────────────────────────────────────────────┐
                      │          ASRS Incident Dataset               │
                      │  (37,301 total: 2021-2025 + Augmented 15-21) │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │                                             │
             ▼                                             ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│     Structured Data     │                   │   Narrative Text Data   │
│       (Event Tags)      │                   │    (Free-Form Text)     │
└────────────┬────────────┘                   └────────────┬────────────┘
             │                                             │
             ├─► Baseline Classifiers (NB, RFC, OVR-SVC)   ├─► Baseline: TF-IDF + Logistic Regression
             │                                             │
             └─► Stacking Ensemble Architecture            ├─► Strategy 1: Encoder-Only ModernBERT
                 • 4 OVR Sub-classifiers (Balanced)        │   • Multiclass-head with weighted Loss
                 • SMOTE-Tomek / SMOTE-ENN Rebalancing     │
                 • Meta-Estimator Decision                 ├─► Strategy 2: Two-Stage Top-K Candidate (RAG)
                                                           │   • ModernBERT Top-K + Custom Guide
                                                           │   • Gemma 4 (E2B-it) LoRA Fine-Tuned
                                                           │
                                                           └─► Strategy 3: Summarization Pipeline
                                                               • GPT OSS-120B (Zero-Shot Summaries)
                                                               • Fine-Tuned ModernBERT on Summaries
```

### 1. Structured Data: Event Tags
Event tags represent categorical hazards labeled during ASRS ingestion.
* **Preprocessing:** One-hot encoded into a ~150-dimensional vector. Near mid-air collision distances were quantized into 5 spatial categories (*extremely close, close, near, far, distant*). Detection descriptions were pruned to top 15 categories.
* **Baseline Models:** Multinomial Naive Bayes, Random Forest Classifier (RFC), and One-vs-Rest (OVR) Support Vector Classifiers (SVC).
* **Stacking Ensemble:** A 2-stage architecture with 4 OVR sub-classifiers trained on balanced sample-size buckets ($n \le 45$, $45 < n \le 300$, $300 < n \le 1000$, $n > 1000$) combined with a meta-classifier using **SMOTE-Tomek** resampling.

### 2. Unstructured Data: Narratives
* **Baseline:** TF-IDF feature extraction + Logistic Regression.
* **Strategy 1: Encoder-Only ModernBERT:** Selected for its 8,192 token context window. Fine-tuned with an inverse-weighted cross-entropy loss function to prevent bias toward majority classes.
* **Strategy 2: Top-K Candidate Decision Pipeline (RAG):** ModernBERT isolates the top-K (3 or 5) candidates. An augmented prompt containing label definitions and common failure modes (generated via Claude 4.6 Sonnet) is evaluated by **Gemma 4 (E2B-it)** fine-tuned via LoRA and Gemini 3 Pro reasoning distillation.
* **Strategy 3: Summarization Pipeline:** **GPT OSS-120B** generates standardized, rhetoric-free incident summaries (zero-shot) to avoid style variance. ModernBERT is then fine-tuned on these clean summaries.

---

## Dataset Overview & Class Distribution

* **Total Samples:** 37,301 incidents
  * **Train Set ($n=22,992$):** Stratified 2021–2025 data + ~10,000 augmented instances (2015–2021) for minority/hard-to-distinguish classes.
  * **Validation Set ($n=4,441$):** Stratified sample (2021–2025).
  * **Test Set ($n=9,868$):** Stratified sample (2021–2025).

| Primary Problem Class | Pre-Augmentation Count | Post-Augmentation Count |
| :--- | :---: | :---: |
| **Logbook Entry** | 2 | 26 |
| **Incorrect / Not Installed / Unavailable Part** | 20 | 112 |
| **MEL (Minimum Equipment List)** | 24 | 148 |
| **Company Policy** | 55 | 1,281 |
| **Software and Automation** | 71 | 71 |
| **Manuals** | 11 | 173 |
| **Staffing** | 11 | 226 |
| **Equipment / Tooling** | 41 | 357 |
| **ATC Equipment / Nav Facility / Buildings** | 133 | 602 |
| **Airspace Structure** | 135 | 664 |
| **Chart or Publication** | 141 | 701 |
| **Environment (Non-Weather-Related)** | 250 | 1,123 |
| **Airport** | 256 | 256 |
| **Weather** | 346 | 1,751 |
| **Procedure** | 1,016 | 5,637 |
| **Aircraft** | 3,474 | 3,474 |
| **Human Factors** | 3,539 | 5,555 |

---

## Experimental Results

### Primary Classification Performance Matrix

| Method | Exact Match Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | **0.66** | 0.26 | 0.38 | 0.28 | 0.67 | **0.66** | **0.66** |
| **Multiclass ModernBERT** | 0.58 | 0.26 | **0.50** | 0.30 | **0.68** | 0.58 | 0.60 |
| **Stacking Classifier (Class Weights)** | 0.53 | 0.24 | 0.40 | 0.26 | 0.66 | 0.53 | 0.57 |
| **Stacking Classifier (SMOTE-Tomek)** | 0.57 | 0.25 | 0.35 | 0.27 | 0.62 | 0.57 | 0.59 |
| **Summarization + ModernBERT** *(Val $n=4.08k$)* | 0.55 | **0.42** | 0.35 | **0.35** | 0.54 | 0.55 | 0.53 |
| **ModernBERT Top-3 + Gemma 4** *(Val $n=937$)* | 0.54 | 0.27 | 0.48 | 0.31 | 0.65 | 0.54 | 0.57 |

### Top-K Accuracy Growth

| Method | Top-1 Accuracy | Top-2 Accuracy | Top-3 Accuracy | Top-4 Accuracy | Top-5 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | **0.66** | **0.84** | **0.91** | **0.94** | **0.96** |
| **Multiclass ModernBERT** | 0.58 | 0.76 | 0.85 | 0.91 | 0.94 |
| **Stacking (SMOTE-Tomek)** | 0.57 | 0.75 | 0.85 | 0.91 | 0.94 |
| **Stacking ($C=1.0$)** | 0.53 | 0.71 | 0.81 | 0.88 | 0.93 |

---

## Discussion & Key Takeaways

1. **The Performance Ceiling:** Advanced neural models (ModernBERT, Gemma 4, GPT OSS-120B) did not surpass the simple **TF-IDF + Logistic Regression** baseline in exact-match accuracy (0.66).
2. **Subjectivity in Ground Truth:** While Top-1 accuracy plateaus around ~60–66%, **Top-3 accuracy quickly exceeds 85–91%**, and Top-5 reaches **96%**. This indicates models consistently identify the correct *candidate cluster*, but forcing a single deterministic choice fails due to label ambiguity.
3. **Class Overlap Examples:**
   * *Drone Proximity Hazards:* Labeled interchangeably as **Environment (Non-Weather-Related)** (external threat) or **Airspace Structure** (regulatory/design conflict).
   * *GPS Jamming:* Frequently assigned to different categories across identical incidents without clear distinction rules.
4. **Conclusion & Future Directions:**
   * Future efforts should focus on **data hygiene and taxonomy reform** rather than increasing model complexity.
   * Recommend involving aviation domain experts to quantify label subjectivity.
   * Explore alternative unsupervised/semi-supervised schemas (e.g., **embedding space clustering**) to discover natural hazard groupings rather than forcing rigid 18-class categories.

---

## References & Prior Art

* **ASRS Dataset:** NASA Aviation Safety Reporting System ([HuggingFace Repository](https://huggingface.co/datasets/sookiemonster/asrs-narratives))
* **Robinson, S. D.:** *Primary Problem Classification from ASRS Narrative Reports* (Multi-Label Classification of Contributing Causal Factors).
* **Tikayat Ray, A. et al.:** *Predicting the Specific Subcategory of Human Factor-related issues: Examining the Potential of Generative Language Models for Aviation Safety Analysis*.
