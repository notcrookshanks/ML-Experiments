# Experiment 4: Feature Scaling and Polynomial Regression

> **Quick Summary:** Compare Linear Regression performance across 4 setups: **Baseline (No Scaling)**, **StandardScaler (Z-Score)**, **MinMaxScaler ([0, 1])**, and **PolynomialFeatures (Degree=2)**. Understand how scaling transforms features and how polynomial terms capture non-linear relationships.

---

## 1. Concept & Intuition

### Why Feature Scaling?
When features have vastly different scales (e.g., `Age` in [18-60] vs `Salary` in [20,000-200,000]), distance-based algorithms or gradient-based optimizers become dominated by the large-scale feature. Scaling brings features onto a common scale.

### Why Polynomial Regression?
Standard linear regression fits a flat straight line: $y = w_1 x + w_0$. If the relationship is curved (e.g., quadratic), standard linear regression underfits. By adding polynomial powers ($x^2$), we fit a parabolic curve while still using linear regression!

---

## 2. Key Formulas to Remember

### 1. StandardScaler (Z-Score Normalization)
$$z = \frac{x - \mu}{\sigma}$$
- Centers data to **mean $\mu = 0$** and scales to **standard deviation $\sigma = 1$**.
- Best when features roughly follow a Gaussian distribution.

### 2. MinMaxScaler (Min-Max Normalization)
$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
- Binds all feature values into the range **$[0, 1]$**.
- Preserves zero values and handles bounded data well.

### 3. Polynomial Features (Degree = 2)
$$[x] \xrightarrow{\text{PolynomialFeatures}} [1, \; x, \; x^2]$$
Model equation:
$$y = w_0 + w_1 x + w_2 x^2$$

---

## 3. Step-by-Step Code Walkthrough

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset
df = pd.DataFrame({
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [15, 25, 35, 45, 55, 65, 75, 85, 95, 100]
})
X = df[['Hours']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. Baseline ---
m1 = LinearRegression().fit(X_train, y_train)
p1 = m1.predict(X_test)

# --- 2. StandardScaler ---
scaler_std = StandardScaler()
X_tr_std = scaler_std.fit_transform(X_train)
X_te_std = scaler_std.transform(X_test)
m2 = LinearRegression().fit(X_tr_std, y_train)
p2 = m2.predict(X_te_std)

# --- 3. MinMaxScaler ---
scaler_mm = MinMaxScaler()
X_tr_mm = scaler_mm.fit_transform(X_train)
X_te_mm = scaler_mm.transform(X_test)
m3 = LinearRegression().fit(X_tr_mm, y_train)
p3 = m3.predict(X_te_mm)

# --- 4. Polynomial Features ---
poly = PolynomialFeatures(degree=2)
X_tr_poly = poly.fit_transform(X_train)
X_te_poly = poly.transform(X_test)
m4 = LinearRegression().fit(X_tr_poly, y_train)
p4 = m4.predict(X_te_poly)
```

### Result Summary Table:
| Setup | MSE | $R^2$ Score | Interpretation |
| :--- | :---: | :---: | :--- |
| **Baseline (No Scaling)** | 2.2341 | 0.9982 | Linear fit on raw values |
| **StandardScaler** | 2.2341 | 0.9982 | Identical metrics (linear shift) |
| **MinMaxScaler** | 2.2341 | 0.9982 | Identical metrics (linear scale) |
| **Polynomial (Degree=2)** | 3.5101 | 0.9971 | Parabolic fit |

---

## 4. Key Functions Cheat Sheet

| Function | Package | Purpose |
| :--- | :--- | :--- |
| `StandardScaler()` | `sklearn.preprocessing` | Centers mean to 0, std dev to 1 |
| `MinMaxScaler()` | `sklearn.preprocessing` | Rescales data into $[0, 1]$ interval |
| `PolynomialFeatures(degree=d)` | `sklearn.preprocessing` | Generates polynomial & interaction features |
| `scaler.fit_transform(X_train)` | `sklearn.preprocessing` | Computes params on train and transforms it |
| `scaler.transform(X_test)` | `sklearn.preprocessing` | Uses train params to transform test (no leakage!) |

> [!IMPORTANT]
> **Data Leakage Rule:** Always `fit_transform` only on `X_train`, then use `transform` on `X_test`. Never call `fit` or `fit_transform` on the test set!

---

## 5. Top Viva / Exam Questions & Answers

**Q1: Why did MSE and $R^2$ stay exactly the same for Baseline, StandardScaler, and MinMaxScaler in simple linear regression?**
> **A:** Ordinary Least Squares (OLS) closed-form solution is invariant to linear transformations ($x' = ax + b$). Scaling simply rescales the learned coefficient $m' = m / a$ and adjusts the intercept $c' = c - mb/a$, producing identical output predictions. However, scaling is critical for Gradient Descent, Ridge/Lasso, KNN, and Neural Networks!

**Q2: Is Polynomial Regression considered a linear model or non-linear model?**
> **A:** It is a **linear model** because it is linear in the parameters ($w_0, w_1, w_2$). Even though the relationship with $x$ is non-linear ($x^2$), the equation $y = w_0 + w_1 x + w_2 x^2$ is solved using standard linear regression algorithms.

**Q3: What risk occurs if the polynomial degree is set too high (e.g., degree = 15)?**
> **A:** **Overfitting**. The curve will try to pass through every single training point, capturing random noise and exhibiting severe variance, which causes high test set error.

