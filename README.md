# beta-kde: Boundary-Corrected Kernel Density Estimation

[![PyPI version](https://badge.fury.io/py/beta-kde.svg)](https://badge.fury.io/py/beta-kde)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/egonmedhatten/beta-kde/actions/workflows/tests.yml/badge.svg)](https://github.com/egonmedhatten/beta-kde/actions)

**Fast, finite-sample boundary correction for data strictly bounded in [0, 1].**

`beta-kde` is a Scikit-learn compatible library for Kernel Density Estimation (KDE) using the Beta kernel approach (Chen, 1999). It fixes the **Boundary Bias** problem inherent in standard Gaussian KDEs, where probability mass "leaks" past the edges of the data (e.g., below 0 or above 1).

This package is the official implementation of the paper:
> **A Fast, Closed-Form Bandwidth Selector for the Beta Kernel Density Estimator**
> *Johan Hallberg Szabadváry (2025)*
> Submitted to Journal of Computational and Graphical Statistics.

## 📊 The Problem vs. The Solution

Standard KDEs smooth data blindly, ignoring bounds. `beta-kde` uses asymmetric Beta kernels that naturally adapt their shape near boundaries to prevent leakage.

![Boundary Bias Comparison](https://raw.githubusercontent.com/egonmedhatten/beta-kde/main/assets/comparison.png)

## 🚀 Key Features

* **Boundary Correction:** Zero leakage. Probability mass stays strictly within the defined bounds.
* **Fast Bandwidth Selection:** Implements the **Beta Reference Rule**, a closed-form $\mathcal{O}(1)$ selector proposed in Szabadváry (2025). It matches the accuracy of expensive Cross-Validation but is **orders of magnitude faster**.
* **Multivariate Support:** Models multivariate bounded data using a **Non-Parametric Beta Copula**.
* **Scikit-learn API:** Drop-in replacement for `KernelDensity`. Fully compatible with `GridSearchCV`, `Pipeline`, and `cross_val_score`.

## 📦 Installation

```bash
pip install beta-kde
```

## ⚡ Quick Start
1. Univariate Data (The Standard Case)
BetaKDE enforces Scikit-learn's 2D input standard (n_samples, n_features).

```python
import numpy as np
from beta_kde import BetaKDE
import matplotlib.pyplot as plt

# 1. Generate bounded data (e.g., ratios or probabilities)
np.random.seed(42)
X = np.random.beta(2, 5, size=(100, 1))  # Must be 2D column vector

# 2. Fit the estimator
# 'beta-reference' is the fast, default rule-of-thumb from the paper
kde = BetaKDE(bandwidth='beta-reference', bounds=(0, 1))
kde.fit(X)

print(f"Selected Bandwidth: {kde.bandwidth_:.4f}")

# 3. Score samples (returns log-likelihood)
log_density = kde.score_samples(np.array([[0.1], [0.5], [0.9]]))

# 4. Plotting convenience
fig, ax = kde.plot()
plt.show()
```
2. Multivariate Data (Copula)
For multidimensional data, BetaKDE fits marginals independently and models dependence using a Copula.
```python
# Generate correlated 2D data
X_2d = np.random.rand(200, 2) 

# Fit (automatically uses Copula for n_features > 1)
kde_multi = BetaKDE(bandwidth='beta-reference')
kde_multi.fit(X_2d)

# Returns log-likelihood of the joint distribution
scores = kde_multi.score_samples(X_2d)
```

## 🆚 Why use beta-kde?
If your data represents percentages, probabilities, or physical constraints (e.g., $x \in [0, 1]$), standard KDEs are mathematically incorrect at the edges.

| Feature | `sklearn.neighbors.KernelDensity` | `beta-kde` |
| :--- | :--- | :--- |
| **Kernel** | Gaussian (Symmetric) | Beta (Asymmetric) |
| **Boundary Handling** | **Biased** (Leaks mass < 0) | **Correct** (Strictly $\ge 0$) |
| **Bandwidth Selection** | Heuristic (Silverman) | **Smart** (Beta Reference Rule) |
| **Multivariate** | Symmetric Gaussian Blob | Flexible Non-Parametric Copula |
| **Speed (Prediction)** | Fast (Tree-based) | Moderate (Exact summation) |

### ⚠️ Important Usage Notes
1. Strict Input Shapes: Input X must be 2D. Use X.reshape(-1, 1) for 1D arrays. This constraint prevents accidental application of univariate estimators to multivariate data.
2. Computational Complexity: This is an exact kernel method. Prediction is $\mathcal{O}(N_{train} \cdot N_{test})$. Recommended for datasets with $N < 50,000$.
3. Bounds: You must specify bounds if your data is not in $[0, 1]$. The estimator handles scaling internally.

### 📚 References
1. Chen, S. X. (1999). Beta kernel estimators for density functions. Computational Statistics & Data Analysis, 31(2), 131-145.
2. Szabadváry, J. H. (2025). A Fast, Closed-Form Bandwidth Selector for the Beta Kernel Density Estimator. Journal of Computational and Graphical Statistics (Submitted).

### Citation
If you use this package in your research, please cite:
```bibtex
@article{szabadvary2025beta,
  title={A Fast, Closed-Form Bandwidth Selector for the Beta Kernel Density Estimator},
  author={Szabadv{\'a}ry, Johan Hallberg},
  journal={Preprint},
  year={2025}
}
```
### License
BSD 3-Clause License