# Network Intrusion Detection — Progressive Dataset Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rosette28/data-science-cyber-final-project/blob/main/notebooks/ids_project.ipynb)

Final project for **Data Science Methods in Cybersecurity** — a reproduction and critical evaluation of:

> Chua, T.-H. & Salam, M. I. (2023). *Evaluation of Machine Learning Algorithms in Network-Based Intrusion Detection Using Progressive Dataset*. **Symmetry**, 15(6), 1251. [https://doi.org/10.3390/sym15061251](https://doi.org/10.3390/sym15061251)

**Authors' original repository:** [tuanhong3498/Evaluation-of-Machine-Learning-Algorithm-in-Network-Based-Intrusion-Detection-System](https://github.com/tuanhong3498/Evaluation-of-Machine-Learning-Algorithm-in-Network-Based-Intrusion-Detection-System)

---

## Project Description

The paper proposes *progressive evaluation* for intrusion detection systems: train six ML models (Decision Tree, Random Forest, SVM, Naive Bayes, ANN, DNN) on **CIC-IDS2017**, then test them on the temporally later **CSE-CIC-IDS2018** to measure how performance survives a real-world distribution shift. The authors report a sharp cross-dataset accuracy drop, attribute it to **overfitting**, and rank SVM and ANN as the most resistant models.

This project rebuilds the full pipeline end-to-end (data loading → EDA → feature engineering → 6-model training with GridSearchCV → multi-metric evaluation → per-attack-type error analysis) and tests the paper's claims with experiments the paper itself did not run:

- an extended metric suite (**MCC, F₂, ROC-AUC**) beyond the paper's Accuracy/Precision/Recall/F1;
- evaluation at **real class prevalence** (84% benign) alongside the paper's 1:1-balanced protocol;
- a matched-class-balance **overfitting vs. concept-drift check** using in-distribution CV stability.

### Key findings

| Paper's claim | Our verdict |
|---|---|
| SVM is the most drift-resistant model | ✅ Reproduces under every evaluation we ran |
| ANN is the second-best model after SVM | ❌ Reversed — DNN beats ANN by 17 accuracy points under the paper's own 1:1 protocol (0.757 vs. 0.590) |
| The cross-dataset drop is caused by overfitting | ❌ Better explained by **concept drift** — every model's k=5 CV std is ≤ 0.32%, so none show classical overfitting signatures |
| 1:1 balancing ("symmetry") gives fair evaluation | ⚠️ It hides base-rate inflation: DT/RF/NB accuracy collapses 22–32 points when re-evaluated at real prevalence |

---

## Repository Structure

```
.
├── notebooks/
│   └── ids_project.ipynb   # Main Colab notebook (§1 Data Loading … §8 Summing It Up)
├── src/                    # Python modules used by the notebook
│   ├── data_loading.py     #   chunked CSV loading, schema alignment, cleaning
│   ├── preprocessing.py    #   relabelling, balancing, derived features, scaling, selection, VIF
│   ├── models.py           #   model factory, GridSearchCV, CV, train/save/load
│   └── evaluation.py       #   metrics, confusion grids, per-attack-type recall, FP/FN examples
├── reports/
│   └── REPORT.md           # Full project report (export to PDF for submission)
├── figures/                # Plots saved by the notebook
├── docs/                   # Paper notes and task tracker
└── requirements.txt
```

---

## Dataset Sources

Both datasets originate from the Canadian Institute for Cybersecurity (CIC/UNB); this project uses the following public Kaggle mirrors of the CICFlowMeter-processed CSVs, rather than downloading directly from UNB/AWS:

| Dataset | Source | Role |
|---------|--------|------|
| CIC-IDS2017 | Kaggle — [chethuhn/network-intrusion-dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) (mirrors UNB's "MachineLearningCVE" CSVs) | Training set (July 2017, ~10% subsample) |
| CSE-CIC-IDS2018 | Kaggle — [ekkykharismadhany/csecicids2018-cleaned](https://www.kaggle.com/datasets/ekkykharismadhany/csecicids2018-cleaned) | Progressive test set (Feb–Mar 2018, ~10% subsample) |

Original UNB pages for reference: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html). `src/data_loading.py` is written for the specific quirks of these two Kaggle files: the CIC-IDS2017 mirror has en-dash-mangled `Web Attack` labels (fixed by `_CIC2017_LABEL_FIX`), and the CSE-CIC-IDS2018 "cleaned" mirror encodes `Label` as integers 1–15 (reversed by `CIC2018_LABEL_MAP`).

The CSVs are **not** committed. Place them in a Google Drive folder:

```
<your Drive>/ids_data/raw/cic2017/*.csv
<your Drive>/ids_data/raw/cic2018/*.csv
```

---

## How to Run (Google Colab)

1. **Open the notebook** — click the Colab badge above.
2. **Run the header cell first** (required at the start of every session). It mounts Google Drive, clones this repo so the `src/` modules are importable, and installs `requirements.txt`.
3. **Set `DATA_DIR`** in the configuration cell — the only line you change:
   ```python
   DATA_DIR = '/content/drive/MyDrive/ids_data/raw/'
   ```
4. **Run all cells** (*Runtime → Run all*). Sections are resume-safe: cleaned data, preprocessed matrices, tuned hyperparameters, and trained models are saved to `DATA_DIR` with `joblib` and reloaded automatically, so a Colab disconnect never forces a full re-run.

Fixed seed (`SEED = 42`) is used throughout; ~10% of each dataset is subsampled at load time to fit free-tier Colab RAM.

---

## Results Snapshot (my run)

Progressive test = models trained on CIC-IDS2017, evaluated on CSE-CIC-IDS2018 at real prevalence (84.1% benign):

| Model | k=5 CV Acc (2017) | Prog. Acc | Prog. Recall | MCC | ROC-AUC |
|-------|------------------:|----------:|-------------:|------:|--------:|
| SVM | 0.9644 | 0.8832 | 0.7448 | 0.6039 | 0.7929 |
| DNN | 0.9684 | 0.8796 | 0.5802 | 0.5345 | **0.9075** |
| RF  | 0.9973 | 0.8626 | 0.1419 | 0.3378 | 0.5741 |
| DT  | 0.9968 | 0.8312 | 0.1433 | 0.1665 | 0.5522 |
| ANN | 0.9666 | 0.8088 | 0.2735 | 0.2067 | 0.8244 |
| NB  | 0.8444 | 0.8084 | 0.0080 | −0.0644 | 0.7111 |

Full analysis, comparison against the paper's Table 7, and error analysis: see the notebook and [`reports/REPORT.md`](reports/REPORT.md).
