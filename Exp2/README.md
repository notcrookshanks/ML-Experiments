# Experiment 2: Simple Linear Regression

> **Quick Summary:** Clean student data (handle duplicates & missing values), fit a **Simple Linear Regression** line ($y = mx + c$) to predict `Final_Score` from `Study_Hours`, evaluate with **MSE** and **$R^2$**, and plot the regression line.

---

## 1. Concept & Intuition

**Simple Linear Regression** models the linear relationship between one independent explanatory variable ($X$) and one dependent target variable ($y$).

Imagine guessing a student's final marks based on their study hours:
- As study hours increase, marks generally follow a straight trend line.
- The algorithm finds the best-fitting straight line by minimizing the sum of squared differences between actual points and line predictions (Ordinary Least Squares - OLS).

---

## 2. Key Formulas to Remember

### Line Equation:
$$y = mx + c$$
- $y$: Predicted Target (`Final_Score`)
- $x$: Input Feature (`Study_Hours`)
- $m$: Slope (`model.coef_[0]`) — how much $y$ changes per unit increase in $x$
- $c$: Y-Intercept (`model.intercept_`) — baseline score when $x = 0$

### Evaluation Metrics:
1. **Mean Squared Error (MSE):** Average of squared errors.
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   *(Lower is better; 0 means perfect prediction)*

2. **$R^2$ Score (Coefficient of Determination):**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   *(Proportion of variance explained by the model; closer to 1.0 is better)*

---

## 3. Step-by-Step Code Walkthrough

### Step 1: Import Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Step 2: Clean the Dataset
```python
df = pd.read_csv('./data_2.csv')

# Drop duplicate rows (always reassign back!)
df = df.drop_duplicates()

# Impute missing values with column means
df = df.fillna(df.mean(numeric_only=True))
```

### Step 3: Feature Matrix & Train-Test Split
```python
# X must be 2D DataFrame [['column']], y is 1D Series ['column']
X = df[['Study_Hours']]
y = df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

### Step 4: Fit Model and Inspect Parameters
```python
model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_
print(f"y = {slope:.2f} * x + {intercept:.2f}")
```

### Step 5: Evaluate on Test Set
```python
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", round(mse, 2))
print("R2 Score:", round(r2, 4))
```

### Step 6: Visualize Regression Line
```python
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Fitted Line')
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.legend()
plt.show()
```

---

## 4. Key Functions Cheat Sheet

| Function | Purpose | Syntax |
| :--- | :--- | :--- |
| `df.drop_duplicates()` | Removes duplicate rows | `df = df.drop_duplicates()` |
| `df.fillna(...)` | Fills missing values | `df.fillna(df.mean(numeric_only=True))` |
| `train_test_split` | Splits into train & test sets | `train_test_split(X, y, test_size=0.2)` |
| `LinearRegression()` | Creates OLS linear model | `model = LinearRegression()` |
| `model.fit(X, y)` | Trains model on data | `model.fit(X_train, y_train)` |
| `model.predict(X)` | Generates predictions | `y_pred = model.predict(X_test)` |
| `model.coef_[0]` | Retrieves line slope ($m$) | `m = model.coef_[0]` |
| `model.intercept_` | Retrieves Y-intercept ($c$) | `c = model.intercept_` |

---

## 5. Top Viva / Exam Questions & Answers

**Q1: Why must $X$ be passed as `df[['Study_Hours']]` (2D) instead of `df['Study_Hours']` (1D)?**
> **A:** Scikit-learn expects feature inputs to be a 2-dimensional array/matrix of shape `(n_samples, n_features)`. `df[['col']]` produces a 2D DataFrame of shape `(N, 1)`, whereas `df['col']` produces a 1D Series of shape `(N,)` which triggers a ValueError.

**Q2: What is the cost function minimized in Linear Regression?**
> **A:** Ordinary Least Squares (OLS) minimizes the Residual Sum of Squares (RSS): $\sum (y_i - \hat{y}_i)^2$.

**Q3: Can an $R^2$ score be negative?**
> **A:** Yes! If the model performs worse than a horizontal line predicting the mean target value ($\bar{y}$), $R^2$ can be negative. This happens when data has extreme outliers or the relationship is strongly non-linear.

