# Experiment 8: Feature Selection using Decision Tree Classifier

> **Quick Summary:** Evaluate how **Feature Selection** impacts classification accuracy by testing a **Decision Tree Classifier** across subsets of informative, redundant, and noise features. Demonstrate that removing noise features improves test performance.

---

## 1. Concept & Intuition

In machine learning, **more features $\ne$ better accuracy**. Datasets often include:
- **Informative features:** Strong correlation with the target class; vital for prediction.
- **Redundant features:** Duplicate or linear combinations of other features (adds no new information).
- **Noise features:** Completely random values that confuse the model, leading to **overfitting**.

**Feature Selection** aims to find the minimal subset of relevant features that maximizes predictive performance and minimizes model complexity.

---

## 2. Dataset Feature Breakdown

Using `make_classification(n_features=6, n_informative=3, n_redundant=1, shuffle=False)`:
- **Features 0, 1, 2:** Informative (Carry real signal)
- **Feature 3:** Redundant (Derived from informative features)
- **Features 4, 5:** Pure Random Noise (Distractions for the tree)

```
┌───────────────────────────────────────┬─────────────────┬──────────────────┐
│ Feature Indices                       │ Category        │ Impact on Model  │
├───────────────────────────────────────┼─────────────────┼──────────────────┤
│ Features [0, 1, 2]                    │ Informative     │ High value       │
│ Feature [3]                           │ Redundant       │ Neutral / Minor  │
│ Features [4, 5]                       │ Noise           │ Harmful / Overfit│
└───────────────────────────────────────┴─────────────────┴──────────────────┘
```

---

## 3. Step-by-Step Code Walkthrough

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Generate dataset with 6 features
X, y = make_classification(
    n_samples=500, n_features=6, n_informative=3,
    n_redundant=1, n_repeated=0, shuffle=False, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define Feature Subsets
feature_subsets = {
    "All 6 Features [0, 1, 2, 3, 4, 5]": [0, 1, 2, 3, 4, 5],
    "First 4 Features (No Noise) [0, 1, 2, 3]": [0, 1, 2, 3],
    "3 Informative Features [0, 1, 2]": [0, 1, 2],
    "First 2 Features [0, 1]": [0, 1],
    "Single Feature [0]": [0]
}

# 3. Train & Evaluate Decision Tree on Each Subset
results = []
for name, features in feature_subsets.items():
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train[:, features], y_train)
    y_pred = model.predict(X_test[:, features])
    acc = accuracy_score(y_test, y_pred)
    results.append({'Subset': name, 'Features': len(features), 'Accuracy': f"{acc*100:.2f}%"})

print(pd.DataFrame(results).to_string(index=False))
```

### Performance Comparison:
| Feature Subset | Feature Count | Accuracy | Analysis |
| :--- | :---: | :---: | :--- |
| **All 6 Features** | 6 | 80.00% | Degraded by noise features [4, 5] |
| **First 4 Features (No Pure Noise)** | 4 | **84.00%** | **Highest Accuracy** — noise eliminated |
| **3 Informative Features** | 3 | 83.00% | High accuracy with only half the features |
| **First 2 Features** | 2 | 83.00% | Decent performance with minimal features |
| **Single Feature** | 1 | 77.00% | Underfitting — lacks enough information |

---

## 4. Key Functions Cheat Sheet

| Function | Package | Purpose |
| :--- | :--- | :--- |
| `make_classification(...)` | `sklearn.datasets` | Generates synthetic dataset with custom informative/redundant features |
| `DecisionTreeClassifier(...)`| `sklearn.tree` | Tree-based classifier using recursive greedy splits |
| `X[:, features]` | `numpy` / slicing | Selects specific columns by index list |
| `accuracy_score(y_true, y_pred)` | `sklearn.metrics` | Calculates classification accuracy percentage |

---

## 5. Top Viva / Exam Questions & Answers

**Q1: What are the main benefits of Feature Selection?**
> **A:**
> 1. **Prevents Overfitting:** Removing noise features improves generalization to unseen data.
> 2. **Faster Training & Inference:** Fewer dimensions reduce memory and compute time.
> 3. **Model Interpretability:** Simpler models are easier to inspect and explain.

**Q2: What is the "Curse of Dimensionality"?**
> **A:** As the number of features (dimensions) grows, the volume of the feature space increases exponentially, making data points extremely sparse. Models need exponentially more data to generalize, and distances between points become uniform, degrading distance-based algorithms.

**Q3: What are the three broad categories of Feature Selection methods?**
> **A:**
> 1. **Filter Methods:** Evaluate statistical properties of features independently of any model (e.g., Variance Threshold, Correlation, Chi-Square).
> 2. **Wrapper Methods:** Use a model to score feature subsets iteratively (e.g., Forward Selection, RFE - Recursive Feature Elimination).
> 3. **Embedded Methods:** Feature selection happens automatically during model training (e.g., Decision Tree feature importances, Lasso L1 regularization).

