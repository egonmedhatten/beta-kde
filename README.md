# Beta Kernel Density Estimation (beta-kde)
Fast, Boundary-Corrected Density Estimation for [0, 1] Data.

`beta-kde` is a Python library for Kernel Density Estimation (KDE) using the Beta kernel approach proposed by Chen (1999). It is designed to be a drop-in replacement for Scikit-learn's density estimators but optimized for bounded data (e.g., probabilities, percentages, rates) where standard Gaussian KDE suffers from boundary bias.

This package serves as the official implementation for the paper:

**A Fast, Closed-Form Bandwidth Selector for the Beta Kernel Density Estimator** Johan Hallberg Szabadváry (2025) Submitted to Journal of Computational and Graphical Statistics

## Features

* **Boundary Correction:** Eliminates boundary bias naturally—no more "leaking" probability mass or artificial hard stops.
* **Scikit-learn API:** Drop-in replacement for `KernelDensity`, fully compatible with pipelines and cross-validation.
* **Custom Support:** While optimized for $[0, 1]$ data (probabilities, rates), it supports **any bounded interval** $[a, b]$ (e.g., 0 to 100) via automatic scaling.
* **Automated Bandwidth Selection:**
    * **MISE Rule (Proposed):** Fast, $\mathcal{O}(1)$ rule-of-thumb from Szabadváry (2025).
    * **LCV / LSCV:** Robust cross-validation methods.

## Installation
```bash
pip install .
```
For development (editable install):
```bash
pip install -e .[dev]
```

## Quickstart
```python
import numpy as np
from beta_kernel import BetaKernelKDE
import matplotlib.pyplot as plt

# 1. Generate bounded data
np.random.seed(42)
data = np.random.beta(a=2, b=5, size=200)

# 2. Fit the estimator
# bandwidth="MISE_rule" is the default fast solver
kde = BetaKernelKDE(bandwidth="MISE_rule")
kde.fit(data)

print(f"Selected Bandwidth: {kde.bandwidth_:.4f}")

# 3. Score new samples (returns log-density)
# The standard score_samples returns un-normalized log-density (asymptotically consistent).
# To get a strictly normalized PDF (integrates to exactly 1.0), set normalized=True.
log_density = kde.score_samples(np.array([0.1, 0.5, 0.9]), normalized=True)

# 4. Plotting convenience method
kde.plot()
plt.show()
```

## Running Tests
```bash
pytest tests/
```
Or use the included helper script if the package is not installed:
```bash
python run_tests.py
```

## References 
* Chen, S. X. (1999). Beta kernel estimators for density functions. Computational Statistics & Data Analysis, 31(2), 131-145.
* Szabadváry, J. H. (2025). A Fast, Closed-Form Bandwidth Selector for the Beta Kernel Density Estimator. Journal of Computational and Graphical Statistics (Submitted).

## Citation
If you use this software in your research, please cite:

TODO: Add citation here!