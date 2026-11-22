# G-Yan! FinTech Credit Risk Analytics — End-to-End CRISP-DM Project

**Repository:** https://github.com/Nykt04/cse315-group4-gyan-credit-risk  
**Course:** CSE315 (45621 - CSP315A/L) Data Science  
**Institution:** Gordon College — College of Computer Studies (BSCS 3-A)  
**Group 4:** Canlas, Julce Lian E. | Edrosalan, Dale Cyrus D. | Ortega, James Dominic T. | Reguindin, Gian Karlo C.  
**Instructor:** Mr. Simon Granil  
**Dataset:** [UCI Default of Credit Card Clients (ID #350)](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) (*N* = 30,000; 22.12% default rate)

---

## Key Results Summary

| Milestone | Highlight |
| :--- | :--- |
| **Phase 1: Leakage-Free Prep** | Removed 35 exact duplicates (*N* = 29,965), flagged 399 disguised missing codes, engineered 8 behavioral risk features, stratified 75/25 split (`22,473` Train / `7,492` Val) **before** fitting mode imputation, `PCA` (2 PCs = **95.68%** bill variance), and `RobustScaler`. |
| **Phase 2A: Descriptive Models** | `seaborn` correlation matrix, `networkx` feature topology (8 nodes, 26 edges), `scikit-learn` `KMeans` (`k=4` Chronic Delinquents = **60.9%** default), and `mlxtend` Apriori (224 rules; top rule `Sep_Delay>=2M & Delinq>=3M -> Default=1` at **73.4% Confidence**, **3.32x Lift**). |
| **Phase 2B: Predictive Models** | 13 logged `scikit-learn` runs across 6 families (`LinearRegression`/`Ridge`, `LDA`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `MLPClassifier`). Champion **GBDT** hits **ROC-AUC = 0.7830, PR-AUC = 0.5664** (`EXP-10` F1 = `0.5394`; CV-calibrated `EXP-10B` **Recall = 72.9%**, exceeding Phase 1's >=70% target). |
| **Phase 3: Portfolio, Report & Pitch** | 8 publication-grade (300 DPI) visualizations, quantitative subgroup fairness & RA 10173 privacy audit, Official Gordon College CCS Comprehensive Final Report PDF, and 10-slide (5-min) Boardroom Pitch Deck. |

---

## Repository Structure

```text
cse315-group4-gyan-credit-risk/
├── README.md
├── requirements.txt
├── PHASE1/
│   └── FINAL COPIES/
│       ├── Group4_G-Yan!_Phase1_BusinessAndDataUnderstanding.pdf
│       ├── Group4_G-Yan!_Phase1_BusinessAndDataUnderstanding.docx
│       ├── Group4_G-Yan!_Phase1_Preprocessing_Colab.ipynb
│       └── Group4_G-Yan!_Phase1_Preprocessing_Pipeline.py
├── PHASE2/
│   └── FINAL COPIES/
│       ├── Group4_G_Yan!_Phase2!_Model_Pipeline.ipynb
│       ├── model_artifacts.pkl
│       ├── Model_Experiment_Log.csv
│       ├── Association_Rules_Log.csv
│       └── KMeans_Experiment_Log.csv
└── PHASE3/
    ├── FINAL COPIES/
    │   ├── Group4_G-Yan!_Phase3_Comprehensive_Final_Report.pdf
    │   ├── Group4_G-Yan!_Phase3_Visualization_Portfolio.ipynb
    │   ├── Group4_G-Yan!_Phase3_Visualization_Portfolio_Deck.pdf
    │   └── Group4_G-Yan!_Phase3_Visualization_Portfolio_Deck.pptx
    ├── phase3_metrics_summary.json
    └── charts/ (fig01..fig08 @ 300 DPI)
```

---

## Reproduction Instructions

```bash
pip install -r requirements.txt
```

1. **Run Phase 1** (`PHASE1/FINAL COPIES/Group4_G-Yan!_Phase1_Preprocessing_Colab.ipynb` or `.py`) to fetch the canonical UCI `.xls` dataset and export the leakage-free train/test/clean CSVs.
2. **Run Phase 2** (`PHASE2/FINAL COPIES/Group4_G_Yan!_Phase2!_Model_Pipeline.ipynb`) to execute all descriptive and predictive models and export `model_artifacts.pkl` + CSV logs.
3. **Run Phase 3** (`PHASE3/FINAL COPIES/Group4_G-Yan!_Phase3_Visualization_Portfolio.ipynb`) to regenerate all 8 high-resolution charts and `phase3_metrics_summary.json`.
