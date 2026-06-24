# ML From Scratch — WNCC, IIT Bombay

A summer-of-code style project to build core machine learning algorithms from
scratch (using mostly just NumPy), before relying on libraries like
scikit-learn. The goal is conceptual understanding over black-box usage.

---

## Week 1 — Python & Core Libraries

**Objectives**
- Learn Python fundamentals
- Learn NumPy, Matplotlib, and Pandas

**Focus**
- Build comfort with Python syntax and data structures
- NumPy: array operations, broadcasting, vectorization
- Matplotlib: basic plotting for visualizing data and results
- Pandas: loading, cleaning, and exploring tabular data

No code submission for this week — purely preparatory learning via W3Schools,
official docs, and SSL slides.

---

## Week 2 — Regression Models & Gradient Descent

**Objectives**
- Learn regression models
- Learn gradient descent algorithms

**Files**
| File | Description |
|------|--------------|
| `linear_regression_ols.py` | Linear Regression solved via the Normal Equation (closed-form OLS) |
| `linear_regression_gd.py` | Linear Regression trained with Batch Gradient Descent |
| `logistic_regression.py` | Binary Logistic Regression trained with Gradient Descent |
| `Week2_Report.docx` | Report covering theory, implementation, and results for all three |

**Key ideas covered**
- Normal Equation: `w = (XᵗX)⁻¹ Xᵗy`
- Gradient Descent update rule: `w := w − α·∇J(w)`
- Sigmoid hypothesis and Binary Cross-Entropy loss for classification
- Feature scaling and cost-convergence tracking

Each script is runnable standalone (`python3 <file>.py`) and generates its own
demo plot using a synthetic dataset.

---

## Week 3 — Decision Trees, K-Means, Anomaly Detection & PCA

**Objectives**
1. Decision Trees and tree ensembles
2. K-Means clustering and anomaly detection
3. PCA — implement face recognition from scratch using PCA (mid-term project)

**Planned files**
| File | Description |
|------|--------------|
| `decision_tree.py` | Decision Tree classifier from scratch |
| `kmeans.py` | K-Means clustering from scratch |
| `anomaly_detection.py` | Gaussian-based anomaly detection |
| `pca.py` | PCA from scratch |
| `face_recognition_pca.py` | Mid-term project: face recognition using PCA |
| `Week3_Report.docx` | Mandatory mid-term project report: approach, key concepts, and code walkthrough |

> Note: the PCA-based face recognition project and its report are **mandatory**
> for successful completion of the SOC.

---

## How to Run

Each week's scripts are self-contained — no external ML libraries (e.g.
scikit-learn) are used for the core algorithm logic, only NumPy/Matplotlib for
math and plotting.

```bash
pip install numpy matplotlib pandas
python3 linear_regression_ols.py
```
