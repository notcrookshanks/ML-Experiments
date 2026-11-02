# Experiment 6: K-Means Clustering & Finding Optimal K

> **Quick Summary:** Cluster unlabeled 2D data using **K-Means**, determine the optimal number of clusters ($K$) using the **Elbow Method (WCSS / Inertia)** and **Silhouette Score**, and visualize the final clusters with their **centroids**.

---

## 1. Concept & Intuition

**K-Means** is a popular partition-based unsupervised clustering algorithm:
1. It partitions $N$ data points into $K$ distinct clusters.
2. It randomly initializes $K$ centroids.
3. **Assignment step:** Assigns each point to its nearest centroid.
4. **Update step:** Recalculates centroids as the arithmetic mean of assigned points.
5. Repeats steps 3-4 until centroids stop moving (convergence).

---

## 2. How to Choose the Optimal $K$?

Because K-Means requires you to specify $K$ upfront, we use two complementary methods:

### 1. The Elbow Method (WCSS / Inertia)
$$\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$
- Measures the total squared distance of samples to their closest cluster center.
- As $K$ increases, WCSS always decreases.
- Look for the **"elbow" point** on the curve — the point after which WCSS decreases much more slowly.

### 2. Silhouette Score
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- $a(i)$: Mean distance to other points in the **same** cluster (intra-cluster cohesion).
- $b(i)$: Mean distance to points in the **nearest neighbor** cluster (inter-cluster separation).
- **Score range:** $[-1, +1]$:
  - $\approx +1$: Sample is well clustered and far from neighboring clusters.
  - $\approx 0$: Sample is on or very close to the decision boundary.
  - $< 0$: Sample was assigned to the wrong cluster.
- **Rule:** The $K$ value with the **highest Silhouette Score** is optimal.

---

## 3. Step-by-Step Code Walkthrough

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Generate 300 points across 4 true clusters
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# 2. Elbow Method: Compute WCSS for K = 1 to 10
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    wcss.append(km.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("K"); plt.ylabel("WCSS (Inertia)"); plt.title("Elbow Method")
plt.show()

# 3. Silhouette Score for K = 2 to 10
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K={k} -> WCSS: {km.inertia_:.2f}, Silhouette: {score:.4f}")

# 4. Fit Final Model with Optimal K=4
final_km = KMeans(n_clusters=4, random_state=42, n_init=10)
final_labels = final_km.fit_predict(X)

# 5. Plot Clusters and Centroids
plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='rainbow', alpha=0.7)
plt.scatter(final_km.cluster_centers_[:, 0], final_km.cluster_centers_[:, 1],
            color='black', marker='X', s=200, label='Centroids')
plt.title("K-Means (Optimal K=4)")
plt.legend()
plt.show()
```

### Typical Results on Blobs Dataset:
- $K=2$: Silhouette = 0.5902
- $K=3$: Silhouette = 0.7569
- **$K=4$: Silhouette = 0.7916 (PEAK / OPTIMAL)**
- $K=5$: Silhouette = 0.6890

---

## 4. Key Functions Cheat Sheet

| Function / Attribute | Package | Purpose |
| :--- | :--- | :--- |
| `KMeans(n_clusters=K)` | `sklearn.cluster` | Initializes K-Means with $K$ clusters |
| `model.inertia_` | `sklearn.cluster` | Returns Within-Cluster Sum of Squares (WCSS) |
| `model.cluster_centers_` | `sklearn.cluster` | Coordinates of learned centroids |
| `silhouette_score(X, labels)`| `sklearn.metrics` | Computes mean Silhouette Coefficient over all samples |

---

## 5. Top Viva / Exam Questions & Answers

**Q1: What does `n_init=10` mean in scikit-learn's `KMeans`?**
> **A:** It means the algorithm will run 10 separate times with different random centroid seeds. Scikit-learn picks the run with the lowest inertia (best clustering), preventing the model from getting stuck in poor local minima.

**Q2: What is the primary limitation of K-Means?**
> **A:**
> 1. You must specify $K$ manually.
> 2. It assumes spherical (circular/globular) clusters of similar size and density.
> 3. It is highly sensitive to outliers.

**Q3: Can Silhouette Score be computed for $K = 1$?**
> **A:** No. Silhouette score requires calculating the distance between a sample and the nearest *neighboring* cluster ($b(i)$). With only 1 cluster, no second cluster exists, so the score is mathematically undefined.

