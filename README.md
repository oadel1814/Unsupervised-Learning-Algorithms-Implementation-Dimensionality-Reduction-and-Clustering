# Unsupervised Learning: Dimensionality Reduction & Clustering

A comprehensive implementation of unsupervised learning algorithms from scratch, including K-Means, GMM-EM, PCA, and Autoencoders, with extensive evaluation and comparative analysis.

## Overview

This project implements six experiments combining dimensionality reduction techniques (PCA, Autoencoder) with clustering algorithms (K-Means, GMM) on the Breast Cancer Wisconsin dataset. All core algorithms are implemented from scratch using NumPy, without relying on scikit-learn for the algorithmic components.

## Core Implementations (From Scratch)

### Clustering Algorithms
- **K-Means**: K-Means++ initialization, random initialization, convergence tracking, inertia history
- **GMM-EM**: Expectation-Maximization algorithm with multiple covariance types (full, tied, diagonal, spherical)

### Dimensionality Reduction
- **PCA**: Principal Component Analysis using SVD, explained variance ratio, component selection
- **Autoencoder**: Deep neural network with configurable hidden layers, activation functions, and regularization

### Evaluation Metrics (10 metrics)
- **Internal Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Inertia
- **External Metrics**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Purity, Homogeneity, Completeness, V-measure

## Experiments

| # | Configuration | Method | Notes |
|---|---|---|---|
| 1 | Original Data | K-Means | Baseline clustering on 30 features |
| 2 | Original Data | GMM | EM algorithm with 4 covariance types |
| 3 | PCA (5 dims) | K-Means | Dimensionality reduction + clustering |
| 4 | PCA (5 dims) | GMM | EM with multiple covariance structures |
| 5 | Autoencoder (5 dims) | K-Means | Non-linear dimensionality reduction |
| 6 | Autoencoder (5 dims) | GMM | Deep learning + EM clustering |

**Total Configurations Tested**: 60+

## Key Features

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

## Dataset

**Breast Cancer Wisconsin Dataset**
- Samples: 569
- Features: 30 (tumor characteristics)
- Classes: 2 (Benign/Malignant)
- Preprocessing: Standardization (mean=0, std=1)

### Key Results
- **Best K-Means Config**: Original data with optimal k=2 (Silhouette: 0.68)
- **Best PCA+K-Means**: 5D reduction with optimized k
- **Best Autoencoder+K-Means**: Non-linear reduction with enhanced clustering

## Visualizations

- **30+ Distinct Plots** across both notebooks
- 2D/3D scatter plots with cluster centroids
- Elbow curves and BIC/AIC comparison charts
- Training loss curves for autoencoders
- Confusion matrices with accuracy metrics
- Heatmap of normalized performance metrics

## Algorithm Details

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


