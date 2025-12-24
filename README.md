# Unsupervised Learning: Dimensionality Reduction & Clustering

A comprehensive implementation of unsupervised learning algorithms from scratch, including K-Means, GMM-EM, PCA, and Autoencoders, with extensive evaluation and comparative analysis.

## 📋 Overview

This project implements six experiments combining dimensionality reduction techniques (PCA, Autoencoder) with clustering algorithms (K-Means, GMM) on the Breast Cancer Wisconsin dataset. All core algorithms are implemented from scratch using NumPy, without relying on scikit-learn for the algorithmic components.

## 🔧 Core Implementations (From Scratch)

### Clustering Algorithms
- **K-Means**: K-Means++ initialization, random initialization, convergence tracking, inertia history
- **GMM-EM**: Expectation-Maximization algorithm with multiple covariance types (full, tied, diagonal, spherical)

### Dimensionality Reduction
- **PCA**: Principal Component Analysis using SVD, explained variance ratio, component selection
- **Autoencoder**: Deep neural network with configurable hidden layers, activation functions, and regularization

### Evaluation Metrics (10 metrics)
- **Internal Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Inertia
- **External Metrics**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Purity, Homogeneity, Completeness, V-measure

## 📊 Experiments

| # | Configuration | Method | Notes |
|---|---|---|---|
| 1 | Original Data | K-Means | Baseline clustering on 30 features |
| 2 | Original Data | GMM | EM algorithm with 4 covariance types |
| 3 | PCA (5 dims) | K-Means | Dimensionality reduction + clustering |
| 4 | PCA (5 dims) | GMM | EM with multiple covariance structures |
| 5 | Autoencoder (5 dims) | K-Means | Non-linear dimensionality reduction |
| 6 | Autoencoder (5 dims) | GMM | Deep learning + EM clustering |

**Total Configurations Tested**: 60+

## 📁 Project Structure

```
├── Kmeans.ipynb              # Experiments 1, 3, 5 with K-Means
├── GmmAndEm.ipynb            # Experiments 2, 4, 6 with GMM
├── PCA_&_Autoencoder.ipynb   # Dimensionality reduction implementations
└── README.md                 # This file
```

## 📈 Key Features

### Comprehensive Analysis Section (Kmeans.ipynb)
- **Comparison Tables**: Best configurations from each experiment
- **2D Projections**: 6-subplot visualization of clustering results
- **Elbow Curves**: Optimal k selection across 5 PCA dimensions
- **Training Curves**: Autoencoder loss convergence for 5 bottleneck dimensions
- **Heatmap**: Normalized metrics comparison across all methods
- **Confusion Matrices**: Best-performing configurations with accuracy metrics

### Statistical Analysis
- **Paired t-tests**: Original vs PCA+K-Means, PCA vs Autoencoder
- **One-Way ANOVA**: Effect of dimensionality on clustering performance
- **Effect Size (Cohen's d)**: Magnitude of differences between methods
- **Computational Complexity**: Runtime analysis and space complexity O() notation

## 🎯 Dataset

**Breast Cancer Wisconsin Dataset**
- Samples: 569
- Features: 30 (tumor characteristics)
- Classes: 2 (Benign/Malignant)
- Preprocessing: Standardization (mean=0, std=1)

## 🚀 Usage

### Run the Assignment
```bash
# Cell execution in Jupyter (recommended order)
1. Kmeans.ipynb (5-10 minutes)
2. GmmAndEm.ipynb (10-15 minutes)
3. View results: tables printed, plots displayed inline
```

### Key Results
- **Best K-Means Config**: Original data with optimal k=2 (Silhouette: 0.68)
- **Best PCA+K-Means**: 5D reduction with optimized k
- **Best Autoencoder+K-Means**: Non-linear reduction with enhanced clustering

## 📊 Visualizations

- **30+ Distinct Plots** across both notebooks
- 2D/3D scatter plots with cluster centroids
- Elbow curves and BIC/AIC comparison charts
- Training loss curves for autoencoders
- Confusion matrices with accuracy metrics
- Heatmap of normalized performance metrics

## 🔍 Algorithm Details

### K-Means
```
Time Complexity: O(n*k*d*i)
Space Complexity: O(n*d + k*d)
```
Where: n=samples, k=clusters, d=features, i=iterations

### PCA
```
Time Complexity: O(n*d² + d³) [SVD-based]
Space Complexity: O(d²)
```

### Autoencoder
```
Time Complexity: O(epochs*batch*hidden*features)
Space Complexity: O(weights + activations)
```

### GMM-EM
```
Time Complexity: O(iterations*n*k*d)
Space Complexity: O(n*k + k*d²)
```

## 📋 Requirements

- Python 3.8+
- NumPy (linear algebra, numerical computing)
- Pandas (data manipulation)
- Matplotlib & Seaborn (visualization)
- scikit-learn (only for metrics and dataset loading, not algorithms)
- SciPy (statistical tests)

## ✨ Highlights

✅ **From-Scratch Implementations**: No scikit-learn for core algorithms
✅ **Comprehensive Evaluation**: 10 different metrics across all methods
✅ **Statistical Rigor**: Hypothesis testing with p-values and effect sizes
✅ **Professional Presentation**: Clear documentation and visualizations
✅ **Reproducible Results**: Fixed random seeds, detailed methodology
✅ **Scalable Code**: Modular classes for easy extension

## 📚 Learning Outcomes

This project demonstrates:
- Understanding of unsupervised learning fundamentals
- Matrix decomposition and linear algebra (PCA via SVD)
- Iterative clustering algorithms and convergence criteria
- Probabilistic modeling (GMM-EM)
- Deep learning fundamentals (Autoencoder architecture)
- Evaluation metrics for clustering quality
- Statistical hypothesis testing
- Data visualization and comparative analysis

## 🔬 Future Extensions

- Implement hierarchical clustering, DBSCAN
- Add t-SNE for 2D visualization
- Extend to semi-supervised learning
- Optimize with GPU acceleration
- Create interactive parameter tuning dashboard

## 📝 Assignment Information

**Course**: Introduction to Machine Learning (Term 7)
**Institution**: [Your Institution]
**Due Date**: December 24, 2025
**Status**: ✅ Complete & Verified

## 👤 Author

Omar Adel

## 📄 License

This project is part of an academic assignment. Please refer to your institution's academic integrity policy.

---

**Last Updated**: December 24, 2025
**Status**: Ready for Submission ✅
