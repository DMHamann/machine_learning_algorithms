# Machine Learning Algorithms

This repository contains Jupyter notebooks implementing key machine learning concepts (sometimes from scratch, sometimes using PyTorch/scikit-learn).


## Notebooks

### 1. **Autoencoder** (`autoencoder.ipynb`)
**Concepts:** Dimensionality reduction, unsupervised learning, denoising

Implementations of three autoencoder variants:
- **Autoencoder for Compression**: Uses a CNN for dimensionality reduction on MNIST images. Learns a compressed latent representation with reconstruction as the objective. 
- **Denoising Autoencoder (DAE)**: Trained on noisy MNIST images to reconstruct clean versions. Robust to input noise.
- **Masked Autoencoder (MAE)**: A portion of input is masked during training. Robust to input masking. 

---

### 2. **Backpropagation** (`backprop.ipynb`)
**Concepts:** Gradient descent, neural network training, optimization

Implements backpropagation from scratch:
- **XOR Gate Learning**: Simple 2-layer network learning the XOR gate; convergence is visualised. 
- **MNIST Classification**: 2-layer network (784 → 30 → 10 neurons) trained on the full MNIST dataset.
- **Loss Tracking**: Shows training loss across 200 epochs.

**Results:** Achieves ~97% accuracy on MNIST with hand-coded backpropagation.

---

### 3. **Canonical Correlation Analysis (CCA)** (`CCA.ipynb`)
**Concepts:** Multivariate analysis, correlation, dimensionality reduction

 Analysis relationships between two sets of variables. Analysis of cross-covariance matrices finds linear combinations that maximise the correlation between paired datasets.

Useful for multi-view learning and multivariate statistics.

---

### 4. **K-Means Clustering** (`k_means.ipynb`)
**Concepts:** Unsupervised learning, clustering, centroid-based clustering

Implements a K-means algorithm for K clusters:
- Iterative centroid updates
- Cluster assignments based on minimum distance (squared)

---

### 5. **Fisher Linear Discriminant** (`fisher.ipynb`)
**Concepts:** Linear classification, dimensionality reduction, supervised learning

Implements Fisher's Linear Discriminant Analysis (FDA) for binary classification:
- Maximises between-class separation
- Minimises within-class variance
- Compares with: naive difference of means, SVM

Implemented on a breast cancer classification (malignant vs. benign) dataset.

---

### 6. **Gaussian Discriminant Analysis (GDA)** (`gaussian_disc.ipynb`)
**Concepts:** Probabilistic classification

Extends Fisher's approach with quadratic discriminants:
- **Linear Discriminant Analysis (LDA)**: Assumes shared covariance across classes
- **Quadratic Discriminant Analysis (QDA)**: Each class has its own covariance matrix

Implemented on wine cultivar classification dataset using 4 compositional features. 

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
