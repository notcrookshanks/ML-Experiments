# Experiment 1: Data Preprocessing & Outlier Detection

> **Quick Summary:** Inspect a dataset for missing values and duplicates, then detect and remove numerical outliers using the **Interquartile Range (IQR)** method, followed by histogram and boxplot visualization.

---

## 1. Concept & Intuition

Real-world datasets are rarely clean. They often contain:
- **Missing values (`NaN`)**: Incomplete data entry.
- **Duplicate rows**: Redundant records skewing sample counts.
- **Outliers**: Extreme values (e.g., Exam score = `180` when max is `100`).

Outliers can distort mean, standard deviation, and machine learning model weights. The **IQR (Interquartile Range)** method is a robust statistical technique that detects values lying far away from the central 50% of the data.

---

## 2. Key Formulas to Remember

$$\text{IQR} = Q_3 - Q_1$$

Where:
- $Q_1$ (25th percentile / First Quartile): 25% of points lie below this value.
- $Q_3$ (75th percentile / Third Quartile): 75% of points lie below this value.
- $\text{IQR}$: The spread of the middle 50% of data.

### Outlier Boundaries:
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

*Any data point $< \text{Lower Bound}$ or $> \text{Upper Bound}$ is flagged as an outlier.*

---

## 3. Step-by-Step Code Walkthrough

### Step 1: Import Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
```

### Step 2: Load the Dataset
```python
df = pd.read_csv('./data_1.csv')
print(df.head())
print("Shape:", df.shape)
```

### Step 3: Check Data Quality
```python
# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Check duplicate rows
print("Duplicate Records:", df.duplicated().sum())
```

### Step 4: Calculate IQR and Filter Outliers
```python
col = "Final_Score"

# Calculate Quartiles
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

# Define Boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Keep only values inside [lower_bound, upper_bound]
df_cleaned = df[df[col].between(lower_bound, upper_bound)]
```

### Step 5: Visualize the Distribution
```python
# Histogram: Shows shape of data distribution
plt.hist(df_cleaned[col], bins=15, color='skyblue', edgecolor='black')
plt.title("Histogram of Final Score")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Box Plot: Confirms no extreme points remain
plt.boxplot(df_cleaned[col].dropna())
plt.title("Box Plot of Final Score")
plt.show()
```

---

## 4. Key Functions Cheat Sheet

| Function | What it does | Example |
| :--- | :--- | :--- |
| `df.isnull().sum()` | Counts `NaN` in each column | `df.isnull().sum()` |
| `df.duplicated().sum()` | Counts identical duplicate rows | `df.duplicated().sum()` |
| `df[col].quantile(0.25)` | Computes 25th percentile ($Q_1$) | `df['Final_Score'].quantile(0.25)` |
| `df[col].between(a, b)` | Returns boolean series if $a \le x \le b$ | `df[df[col].between(low, up)]` |
| `plt.hist(...)` | Draws frequency distribution | `plt.hist(df[col], bins=15)` |
| `plt.boxplot(...)` | Draws box-and-whisker plot | `plt.boxplot(df[col])` |

---

## 5. Top Viva / Exam Questions & Answers

**Q1: What does IQR stand for and why is it preferred over Z-score for outlier detection?**
> **A:** IQR stands for Interquartile Range ($Q_3 - Q_1$). It is non-parametric and based on medians and percentiles, making it resistant to extreme outliers. The Z-score depends on the mean and standard deviation, which are themselves heavily distorted by extreme outliers.

**Q2: Why is the multiplier $1.5$ used in $1.5 \times \text{IQR}$?**
> **A:** Established by John Tukey, $1.5 \times \text{IQR}$ corresponds roughly to $\pm 2.7\sigma$ in a normal distribution, filtering out approximately the extreme 0.7% of points without discarding legitimate data.

**Q3: How does a Box Plot represent data?**
> **A:** The box represents the middle 50% ($Q_1$ to $Q_3$), the line inside the box is the median ($Q_2$), the whiskers extend to the minimum and maximum within the bounds, and points beyond the whiskers are plotted as individual outlier dots.

