import sys
import os

# Add the ../src directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.neighbors import KernelDensity
from beta_kde import BetaKDE

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

# 1. Generate Data (Beta(2, 12) - Skewed towards 0)
np.random.seed(42)
data = np.random.beta(2, 12, size=(5000, 1))

# 2. Standard Gaussian KDE (Silverman Rule)
kde_gauss = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(data)

# 3. Beta KDE (Proposed Rule)
kde_beta = BetaKDE(bandwidth='beta-reference').fit(data)

# 4. Evaluate on a grid
x_grid = np.linspace(0, 1, 1000).reshape(-1, 1)
pdf_true = beta.pdf(x_grid, 2, 12)
pdf_gauss = np.exp(kde_gauss.score_samples(x_grid))
pdf_beta = np.exp(kde_beta.score_samples(x_grid))

# 5. Plot
plt.figure()
plt.hist(data, bins=50, density=True, alpha=0.2, color='gray', label='Data Histogram')
plt.plot(x_grid, pdf_true, 'k-', lw=1.5, label='True Density')
plt.plot(x_grid, pdf_gauss, 'g--', lw=2, label='Gaussian KDE (Boundary Bias)')
plt.plot(x_grid, pdf_beta, 'r-.', lw=2, label='Beta KDE (Correct)')

plt.title("Boundary Bias Comparison")
plt.xlabel("x")
plt.ylabel("Density")
plt.xlim(0, 0.6)  # Zoom in to show the boundary clearly
plt.legend()
plt.grid(True, alpha=0.3)

# Save to the assets folder
output_path = "assets/comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()