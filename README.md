# Beta Kernel Density Estimation (beta-kernel)
Fast, Boundary-Corrected Density Estimation for [0, 1] Data.

```beta-kernel``` is a Python library for Kernel Density Estimation (KDE) using the Beta kernel approach proposed by Chen (1999). It is designed to be a drop-in replacement for Scikit-learn's density estimators but optimized for bounded data (e.g., probabilities, percentages, rates) where standard Gaussian KDE suffers from boundary bias.

## Features
* **Boundary Correction:** Eliminates boundary bias at x=0 and x=1 naturally.

* **Scikit-learn API:** Fully compatible with sklearn pipelines and cross-validation.

* **Automated Bandwidth Selection:**

    * **MISE Rule:** Fast rule-of-thumb (Chen, 1999).

    * **LCV:** Likelihood Cross-Validation.

    * **LSCV:** Least-Squares Cross-Validation.

## Installation
```
pip install .
```
For development (editable install):
```
pip install -e .[dev]
```

## Quickstart
```
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
```
pytest tests/
```

## References 
* Chen, S. X. (1999). Beta kernel estimators for density functions. Computational Statistics & Data Analysis, 31(2), 131-145.

## Citation
If you use this software in your research, please cite:

TODO: Add citation here!