# Experiment 5: DBSCAN Clustering

> **Quick Summary:** Cluster synthetic 2D data with added noise using **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise), standardize features, tune **$\epsilon$** and **`min_samples`**, separate clusters from noise (labeled `-1`), and compute the **Silhouette Score**.

---

## 1. Concept & Intuition

Unlike K-Means (which assumes round, spherical clusters and forces every point into a cluster), **DBSCAN** groups points based on **density**:
- Dense regions are identified as clusters.
- Sparse isolated regions are marked as **noise / outliers** (label `-1`).
- DBSCAN can discover arbitrary shapes (e.g. rings, crescents, moons) and does **not** require specifying the number of clusters $K$ in advance.

---

## 2. Key Hyperparameters & Point Types

### The Two Parameters:
1. **$\epsilon$ (`eps`)**: The radius of the neighborhood around each point.
2. **`min_samples` (MinPts)**: The minimum number of points required within distance $\epsilon$ to define a dense region.

```
       Core Point                  Border Point                  Noise Point
    (>= min_samples              (< min_samples within        (< min_samples, not
     within distance ε)           ε, but near a core point)    near any core point)
           ●                           ○                             x
        ●  ●  ●                         ●  ●
           ●
```

### The Three Types of Points:
- **Core Point:** Has at least `min_samples` within distance $\epsilon$.
- **Border Point:** Has fewer than `min_samples` within $\epsilon$, but falls within the neighborhood of a Core Point.
- **Noise Point:** Neither a Core nor a Border point. DBSCAN assigns it the label **`-1`**.

---

## 3. Step-by-Step Code Walkthrough

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Generate 3 dense clusters + 50 uniform noise points
X_blobs, y_blobs = make_blobs(
    n_samples=[200, 200, 200], centers=[(-5, -2), (0, 0), (5, 4)],
    cluster_std=[0.4, 0.8, 1.5], random_state=42
)
noise = np.random.RandomState(42).uniform(-10, 10, size=(50, 2))
X = np.vstack([X_blobs, noise])

# 2. Standardize features (DBSCAN relies on Euclidean distances)
X_scaled = StandardScaler().fit_transform(X)

# 3. Fit DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 4. Count clusters and noise (-1)
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = np.sum(labels == -1)
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")

# 5. Evaluate Silhouette Score (on clustered points only)
mask = (labels != -1)
score = silhouette_score(X_scaled[mask], labels[mask])
print(f"Silhouette Score (excluding noise): {score:.4f}")

# 6. Scatter Plot
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='rainbow', s=30)
plt.title(f"DBSCAN: {n_clusters} Clusters, {n_noise} Noise Points")
plt.colorbar(label="Cluster (-1 = Noise)")
plt.show()
```

---

## 4. Key Functions Cheat Sheet

| Function | Package | Purpose |
| :--- | :--- | :--- |
| `make_blobs(...)` | `sklearn.datasets` | Generates isotropic Gaussian blobs for clustering |
| `StandardScaler()` | `sklearn.preprocessing` | Standardizes data to zero mean and unit variance |
| `DBSCAN(eps, min_samples)` | `sklearn.cluster` | Density-based clustering algorithm |
| `labels = model.fit_predict(X)` | `sklearn.cluster` | Fits model and returns cluster IDs (-1 for noise) |
| `silhouette_score(X, labels)` | `sklearn.metrics` | Measures cluster separation and cohesion |

---

## 5. Top Viva / Exam Questions & Answers

**Q1: How does DBSCAN handle noise and outliers compared to K-Means?**
> **A:** K-Means forces every single point into one of the $K$ clusters, which pulls centroids towards outliers. DBSCAN naturally identifies low-density outliers and assigns them a distinct label of `-1` without corrupting valid clusters.

**Q2: What happens if `eps` ($\epsilon$) is chosen too small or too large?**
> **A:**
> - If $\epsilon$ is **too small**: Most points will fail to meet density requirements, and almost everything is labeled as noise (`-1`).
> - If $\epsilon$ is **too large**: Different clusters merge together into one giant single cluster.

**Q3: Why is `StandardScaler` essential before running DBSCAN?**
> **A:** DBSCAN uses Euclidean distance ($\sqrt{\sum (x_i - y_i)^2}$) with a fixed threshold $\epsilon$. If one feature has values from $1$ to $1000$ and another from $0.1$ to $1$, the distance is completely dominated by the first feature, rendering $\epsilon$ meaningless for the other feature.

