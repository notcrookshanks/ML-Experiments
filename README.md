# Machine Learning Practicals Guide 🚀

A clean, readable, and beginner-friendly repository containing complete implementations, visualizations, and revision guides for all 8 Machine Learning laboratory experiments.

---

## 📚 Experiments Overview & Master Table

| # | Experiment Title | Category | Key Classes / Algorithms | Evaluation Metrics | Quick Guide |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **1** | [**Data Preprocessing & Outlier Detection**](file:///Users/bala/ML-Practicals/Exp1/README.md) | Data Cleaning | Pandas, IQR Method, Matplotlib | Missing count, IQR bounds | [Exp1 Guide](file:///Users/bala/ML-Practicals/Exp1/README.md) |
| **2** | [**Simple Linear Regression**](file:///Users/bala/ML-Practicals/Exp2/README.md) | Supervised Regression | `LinearRegression`, `train_test_split` | MSE, $R^2$ Score | [Exp2 Guide](file:///Users/bala/ML-Practicals/Exp2/README.md) |
| **3** | [**Categorical Naive Bayes**](file:///Users/bala/ML-Practicals/Exp3/README.md) | Supervised Classification | `CategoricalNB`, `OrdinalEncoder`, `LabelEncoder` | Accuracy, Precision, Recall | [Exp3 Guide](file:///Users/bala/ML-Practicals/Exp3/README.md) |
| **4** | [**Feature Scaling & Polynomial Regression**](file:///Users/bala/ML-Practicals/Exp4/README.md) | Feature Transformation | `StandardScaler`, `MinMaxScaler`, `PolynomialFeatures` | MSE, $R^2$ Score | [Exp4 Guide](file:///Users/bala/ML-Practicals/Exp4/README.md) |
| **5** | [**DBSCAN Clustering**](file:///Users/bala/ML-Practicals/Exp5/README.md) | Unsupervised Clustering | `DBSCAN`, `StandardScaler`, `make_blobs` | Silhouette Score, Noise Count | [Exp5 Guide](file:///Users/bala/ML-Practicals/Exp5/README.md) |
| **6** | [**K-Means & Optimal K (Elbow Method)**](file:///Users/bala/ML-Practicals/Exp6/README.md) | Unsupervised Clustering | `KMeans`, `silhouette_score`, `make_blobs` | WCSS (Inertia), Silhouette Score | [Exp6 Guide](file:///Users/bala/ML-Practicals/Exp6/README.md) |
| **7** | [**Semi-Supervised Learning (Self-Training)**](file:///Users/bala/ML-Practicals/Exp7/README.md) | Semi-Supervised | `SelfTrainingClassifier`, `LogisticRegression` | Accuracy, Pseudo-label count | [Exp7 Guide](file:///Users/bala/ML-Practicals/Exp7/README.md) |
| **8** | [**Feature Selection (Decision Tree)**](file:///Users/bala/ML-Practicals/Exp8/README.md) | Feature Engineering | `DecisionTreeClassifier`, `make_classification` | Accuracy across feature subsets | [Exp8 Guide](file:///Users/bala/ML-Practicals/Exp8/README.md) |

---

## 🧠 60-Second Memory Map for Viva / Exams

| Exp | Core Formula / Idea | Crucial Code Snippet |
| :-: | :--- | :--- |
| **1** | $\text{IQR} = Q_3 - Q_1$; Bounds = $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$ | `df = df[df[col].between(lower, upper)]` |
| **2** | Line equation: $y = mx + c$; Minimize RSS | `model = LinearRegression(); model.fit(X_tr, y_tr)` |
| **3** | Bayes Rule: $P(Y \mid X) = \frac{P(X \mid Y)P(Y)}{P(X)}$ | `X_enc = OrdinalEncoder().fit_transform(X)` |
| **4** | $z = \frac{x - \mu}{\sigma}$ (Std) vs $\frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (MinMax) | `scaler.fit_transform(X_train); scaler.transform(X_test)` |
| **5** | Density reachability with $\epsilon$ and `min_samples`; noise = `-1` | `DBSCAN(eps=0.3, min_samples=5).fit_predict(X)` |
| **6** | Elbow at WCSS curve; peak Silhouette Score | `KMeans(n_clusters=k).fit(X); model.inertia_` |
| **7** | Pseudo-label high-confidence ($p \ge \text{threshold}$) unlabeled points | `SelfTrainingClassifier(LogisticRegression(), threshold=0.8)` |
| **8** | Remove noise/redundant features to boost test accuracy | `model.fit(X_train[:, features], y_train)` |

---

## 🛠️ Setup & Environment

All dependencies are pinned in [`requirements.txt`](file:///Users/bala/ML-Practicals/requirements.txt).

### Running in Virtual Environment:
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run any experiment notebook in Jupyter or VS Code / Cursor:
# Simply open Exp<N>/index.ipynb
```

Each experiment directory contains:
- `index.ipynb`: Clean, fully executed Jupyter Notebook with step-by-step comments and pre-rendered plots.
- `README.md`: Concise, exam-ready revision guide with formulas, code walkthroughs, and viva Q&A.

