# Experiment 3: Categorical Naive Bayes Classification

> **Quick Summary:** Build a **Naive Bayes** classifier on categorical text data (`Weather` & `Temperature`) to predict whether to play (`Play`), encode categories with `OrdinalEncoder` and `LabelEncoder`, evaluate accuracy, and classify a new observation.

---

## 1. Concept & Intuition

**Naive Bayes** is a probabilistic classification algorithm based on **Bayes' Theorem**. It is called *"Naive"* because it makes the simplifying assumption that all input features are mutually independent given the class label.

- **Bayes' Rule**:
  $$P(\text{Play} \mid \text{Weather}, \text{Temp}) = \frac{P(\text{Weather}, \text{Temp} \mid \text{Play}) \times P(\text{Play})}{P(\text{Weather}, \text{Temp})}$$
- **Conditional Independence Assumption**:
  $$P(\text{Weather}, \text{Temp} \mid \text{Play}) = P(\text{Weather} \mid \text{Play}) \times P(\text{Temp} \mid \text{Play})$$

For discrete categorical counts, scikit-learn provides **`CategoricalNB`**, which computes category frequency tables for each feature.

---

## 2. Key Encoders to Remember

Machine learning models only understand numbers, so text categories must be converted:

1. **`OrdinalEncoder` (for Features $X$)**:
   - Converts multiple categorical feature columns into numeric integers ($0, 1, 2\dots$).
   - Example: `Sunny -> 2, Rainy -> 1, Cloudy -> 0`.
2. **`LabelEncoder` (for Target $y$)**:
   - Converts 1D class labels into integer targets.
   - Example: `'No' -> 0, 'Yes' -> 1`.
   - Allows using `inverse_transform()` to decode predicted numbers back to human-readable strings!

---

## 3. Step-by-Step Code Walkthrough

### Step 1: Import Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report
```

### Step 2: Define Categorical Dataset
```python
df = pd.DataFrame({
    'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy',
                'Cloudy', 'Sunny', 'Rainy', 'Cloudy', 'Sunny'],
    'Temperature': ['Hot', 'Cool', 'Mild', 'Mild', 'Hot',
                    'Cool', 'Cool', 'Mild', 'Hot', 'Hot'],
    'Play': ['No', 'Yes', 'Yes', 'No', 'Yes',
             'Yes', 'Yes', 'Yes', 'Yes', 'No']
})
```

### Step 3: Encode Categories
```python
X = df[['Weather', 'Temperature']]
y = df['Play']

# Encode features
feature_encoder = OrdinalEncoder()
X_encoded = feature_encoder.fit_transform(X)

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
```

### Step 4: Split & Train CategoricalNB
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42
)

model = CategoricalNB()
model.fit(X_train, y_train)
```

### Step 5: Evaluate
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
```

### Step 6: Predict on Unseen Scenario
```python
sample = pd.DataFrame({'Weather': ['Sunny'], 'Temperature': ['Cool']})
sample_encoded = feature_encoder.transform(sample)

pred_num = model.predict(sample_encoded)
pred_label = target_encoder.inverse_transform(pred_num)
print("Prediction for Sunny & Cool:", pred_label[0])
```

---

## 4. Key Functions Cheat Sheet

| Function | Package | Purpose |
| :--- | :--- | :--- |
| `OrdinalEncoder()` | `sklearn.preprocessing` | Encodes 2D feature DataFrame into integer arrays |
| `LabelEncoder()` | `sklearn.preprocessing` | Encodes 1D target Series into 0/1 integers |
| `CategoricalNB()` | `sklearn.naive_bayes` | Naive Bayes designed for categorical/discrete data |
| `encoder.inverse_transform()` | `sklearn.preprocessing` | Translates numeric predictions back to text strings |
| `accuracy_score()` | `sklearn.metrics` | Fraction of correct classifications ($\frac{\text{Correct}}{\text{Total}}$) |

---

## 5. Top Viva / Exam Questions & Answers

**Q1: Why is Naive Bayes called "naive"?**
> **A:** It assumes that all predictor features are strictly independent of each other given the target class. Even when features are correlated in reality, Naive Bayes often still performs surprisingly well.

**Q2: What is the difference between `CategoricalNB`, `GaussianNB`, and `MultinomialNB`?**
> **A:**
> - `CategoricalNB`: Features are discrete categories (e.g. Weather: Sunny/Rainy/Cloudy).
> - `GaussianNB`: Features are continuous numerical values following a normal distribution.
> - `MultinomialNB`: Features are discrete word counts / frequency vectors (used in text/NLP).

**Q3: What is Laplace Smoothing in Naive Bayes?**
> **A:** If a feature category never appeared with a class in training, its probability would be 0, causing the entire product $P(X|Y)$ to collapse to 0 (Zero Probability Problem). Laplace smoothing adds a small positive constant $\alpha = 1.0$ to all counts to prevent zero probability.

