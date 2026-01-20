# Reverse-SMOTE (Reverse Nearest Neighbor–Integrated SMOTE)

Implementation and experimental scripts for the paper:

**An Oversampling Technique by Integrating Reverse-Nearest Neighbor in SMOTE: Reverse-SMOTE**  
Riju Das, Saroj Kr. Biswas, Debashree Devi, Biswajit Sarma  
**ICOSEC 2020** · **DOI:** 10.1109/ICOSEC49089.2020.9215387

---

## 📌 Overview

Class imbalance is a common problem in real-world datasets (fraud detection, medical diagnosis, anomaly detection, etc.).  
Traditional **SMOTE** generates synthetic minority samples using **k-nearest neighbors (kNN)** within the minority class, but it can suffer from **overfitting** when synthetic points are generated without considering how the **majority** class is distributed.

**Reverse-SMOTE (R-SMOTE)** addresses this by integrating the concept of **Reverse Nearest Neighbors (RNN)** into SMOTE.  
Instead of using only kNN neighbors, the algorithm uses **reverse-nearest-neighborhoods** to create synthetic samples in a way that better reflects the data distribution and reduces noisy/overfit synthetic generation.

---

## ✨ Key Idea (What makes Reverse-SMOTE different?)

### SMOTE (standard)
- Picks a minority sample
- Finds **k nearest minority neighbors**
- Generates synthetic points between them

### Reverse-SMOTE (proposed)
- Identifies a **significant set** of minority samples
- Finds their **Reverse Nearest Neighbors (RNN)**
  - i.e., points for which a minority sample appears in their neighbor list
- Generates synthetic points **along the line joining minority samples and their reverse neighbors**

This makes the oversampling process more **adaptive**, often improving minority-class learning while avoiding excessive synthetic clustering.

---

## 📂 Repository Contents

Key files included in this repository:

- `Reverse_SMOTE_module1.py` — Core module for Reverse-SMOTE logic
- `KNN_proposed.py` — kNN classifier used in experiments/evaluation
- `DatasetRep.py` — Dataset representation utilities
- `Reverse_SMOTE-call.ipynb` — Example notebook demonstrating usage
- Dataset experiment scripts:
  - `Reverse_SMOTE_pima.py`
  - `Reverse_SMOTE_ecoli3.py`
  - `Reverse_SMOTE_page_block.py`
  - `Reverse_SMOTE_breast_cancer.py`
- ROC generation scripts:
  - `ROC_pima.py`, `ROC_ecoli3.py`, `ROC_page_block.py`
- ROC plots:
  - `ROC_pima.png`
  - `ROC_ecoli3.png`
  - `ROC_Pageblock.png`

---


 
