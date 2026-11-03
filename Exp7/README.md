# Experiment 7: Semi-Supervised Learning (Self-Training Classifier)

> **Quick Summary:** Address a real-world scenario where **only 20% of data is labeled** and **80% is unlabeled**. Train a **`SelfTrainingClassifier`** that iteratively predicts high-confidence **pseudo-labels** for unlabeled data, improving accuracy over a purely supervised baseline.

---

## 1. Concept & Intuition

In industry, acquiring labeled data is slow and expensive (requires human annotators, medical doctors, domain experts), while unlabeled data is plentiful and cheap.

**Semi-Supervised Learning** bridges this gap:
- **Supervised Learning**: Needs 100% labeled data.
- **Unsupervised Learning**: Uses 0% labeled data (clustering).
- **Semi-Supervised Learning**: Uses a small labeled pool + a large unlabeled pool to achieve high accuracy.

### How Does Self-Training Work?
```
  [20% Labeled Data] ──► Train Base Classifier (Logistic Regression)
                               │
                               ▼
  [80% Unlabeled Data] ──► Predict Class Probabilities
                               │
               Is Confidence >= Threshold (0.8)?
                   ├── YES ──► Assign "Pseudo-Label" & add to Labeled Set
                   └── NO  ──► Keep Unlabeled for next round
                               │
                               ▼
        Iterate until max_iter or no new points pass threshold
```

---

## 2. Scikit-learn Convention for Unlabeled Data

In `scikit-learn`:
- Labeled classes: $0, 1, 2\dots$
- **Unlabeled samples: Must be assigned the value `-1`!**

When `SelfTrainingClassifier.fit(X, y)` runs, it automatically treats all samples where $y = -1$ as unlabeled candidates for pseudo-labeling.

---

## 3. Step-by-Step Code Walkthrough

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score

# 1. Generate classification dataset
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                           n_redundant=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Simulate 80% unlabeled data by marking with -1
rng = np.random.RandomState(42)
y_semi = y_train.copy()
unlabeled_idx = rng.choice(len(y_train), size=int(0.8 * len(y_train)), replace=False)
y_semi[unlabeled_idx] = -1

labeled_mask = (y_semi != -1)
print(f"Labeled: {np.sum(labeled_mask)}, Unlabeled: {np.sum(~labeled_mask)}")

# 3. Supervised Baseline (Trained on only the 20% labeled data)
baseline = LogisticRegression()
baseline.fit(X_train[labeled_mask], y_semi[labeled_mask])
base_acc = accuracy_score(y_test, baseline.predict(X_test))

# 4. Self-Training Classifier (Semi-Supervised)
self_training = SelfTrainingClassifier(
    estimator=LogisticRegression(),
    threshold=0.8,    # Confidence >= 80% required to pseudo-label
    max_iter=10
)
self_training.fit(X_train, y_semi)
self_acc = accuracy_score(y_test, self_training.predict(X_test))

# Count pseudo-labeled points
pseudo_count = np.sum(self_training.transduction_ != -1) - np.sum(labeled_mask)

print("Supervised Baseline Accuracy (20% data) :", f"{base_acc * 100:.2f}%")
print("Self-Training Accuracy (Semi-Supervised):", f"{self_acc * 100:.2f}%")
print(f"Pseudo-labeled samples added: {pseudo_count}")
```

### Result Comparison:
- **Baseline (20% Labeled Data):** 88.00%
- **Self-Training (Added 281 pseudo-labeled points):** 89.00%
- **Accuracy Gain:** +1.00% without manually labeling a single extra point!

---

## 4. Key Functions Cheat Sheet

| Function / Parameter | Purpose | Details |
| :--- | :--- | :--- |
| `SelfTrainingClassifier` | Semi-supervised meta-estimator | Wraps any supervised classifier that has `predict_proba` |
| `threshold=0.8` | Minimum probability required | Only pseudo-labels predictions with $\ge 80\%$ confidence |
| `max_iter=10` | Maximum iterative cycles | Stops after 10 loops or when no new points exceed threshold |
| `model.transduction_` | Final assigned labels | Contains original labels + newly assigned pseudo-labels |

---

## 5. Top Viva / Exam Questions & Answers

**Q1: What is a "Pseudo-Label"?**
> **A:** An artificial label predicted by the model for an unlabeled sample that is accepted as "ground truth" and added to future training rounds because the model's prediction confidence exceeded the threshold.

**Q2: What is "Confirmation Bias" (Error Propagation) in Self-Training?**
> **A:** If the initial base model makes an incorrect prediction with high confidence, that wrong label is added to the training set as a pseudo-label. In subsequent rounds, the model reinforces its own mistake, degrading overall performance.

**Q3: Which classifiers can be used as the base estimator inside `SelfTrainingClassifier`?**
> **A:** Any classifier that supports the `predict_proba` method (e.g. `LogisticRegression`, `RandomForestClassifier`, `SVC(probability=True)`), because the threshold is evaluated on class probability values.

