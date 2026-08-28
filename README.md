# Machine Learning Algorithms

A collection of some of my implementations of fundamental machine learning algorithms.
## Overview

This repository contains Jupyter notebooks implementing key machine learning concepts from scratch and using PyTorch/scikit-learn. Each notebook is self-contained with theory, implementation, and visualization.


## Notebooks

### 1. **Autoencoder** (`autoencoder.ipynb`)
**Concepts:** Dimensionality reduction, unsupervised learning, denoising

Implementations of three autoencoder variants:
- **Autoencoder for Compression**: Uses convolutional layers for dimensionality reduction on MNIST images. Learns a compressed latent representation with reconstruction as the objective.
- **Denoising Autoencoder (DAE)**: Trained on noisy MNIST images to reconstruct clean versions. Demonstrates robustness to input noise.
- **Masked Autoencoder (MAE)**: Variation where a portion of input is masked during training.

**Key Results:** Visualizes original vs. reconstructed images at different compression levels.

---

### 2. **Backpropagation** (`backprop.ipynb`)
**Concepts:** Gradient descent, neural network training, optimization

Implements backpropagation from scratch according to a linked book by Rojas:
- **XOR Gate Learning**: Simple 2-layer network learning the XOR truth table, visualizing convergence per input.
- **MNIST Classification**: 2-layer network (784 → 30 → 10 neurons) trained on full MNIST dataset with sigmoid activation.
- **Loss Tracking**: Shows training dynamics across 200 epochs with loss convergence visualization.

**Key Results:** Achieves ~97% accuracy on MNIST with hand-coded backpropagation, demonstrating algorithm correctness.

---

### 3. **Canonical Correlation Analysis (CCA)** (`CCA.ipynb`)
**Concepts:** Multivariate analysis, correlation, dimensionality reduction

Analyzes relationships between two sets of variables. CCA finds linear combinations that maximize correlation between paired datasets.

**Application:** Useful for understanding relationships in multi-view data or comparing different feature sets measuring the same phenomenon.

---

### 4. **K-Means Clustering** (`k_means.ipynb`)
**Concepts:** Unsupervised learning, clustering, centroid-based methods

Implements the K-means algorithm for partitioning data into K clusters:
- Iterative centroid updates
- Cluster assignments based on minimum distance
- Convergence criteria

**Application:** Image compression, document clustering, customer segmentation.

---

### 5. **Fisher Linear Discriminant** (`fisher.ipynb`)
**Concepts:** Linear classification, dimensionality reduction, supervised learning

Implements Fisher's Linear Discriminant Analysis (FDA) for binary classification:
- Maximizes between-class separation
- Minimizes within-class variance
- Compares with: naive difference of means, SVM

**Dataset:** Breast cancer classification (malignant vs. benign)

**Key Insight:** Different discriminants extract different aspects of data—FDA balances class separation with low within-class variance.

---

### 6. **Gaussian Discriminant Analysis (QDA)** (`gaussian_disc.ipynb`)
**Concepts:** Probabilistic models, Bayesian classification, quadratic decision boundaries

Extends Fisher's approach with quadratic discriminants:
- **Linear Discriminant Analysis (LDA)**: Assumes shared covariance across classes
- **Quadratic Discriminant Analysis (QDA)**: Each class has its own covariance matrix

**Dataset:** Wine cultivar classification using 4 compositional features (Mg, phenols, flavanoids, etc.)

**Key Results:** 
- Fisher LDA AUROC: 0.942
- QDA AUROC: 0.988

QDA outperforms LDA when classes have different covariance structures.


### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/machine_learning_algorithms.git
cd machine_learning_algorithms

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
