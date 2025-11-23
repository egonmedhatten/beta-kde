import sys
import os

# Add the ../src directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_path)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function
from beta_kde import BetaKDE

os.makedirs("assets", exist_ok=True)

# 1. Generate "Nice" Bimodal Data (Logit-Normal Mixture)
np.random.seed(42)
n = 3000

# Component 1: Lower-left cluster
mean1 = [-1.5, -1.5]
cov1 = [[0.8, 0.4], [0.4, 0.8]]
data1 = np.random.multivariate_normal(mean1, cov1, size=n // 2)

# Component 2: Upper-right cluster
mean2 = [1.5, 1.5]
cov2 = [[0.5, -0.2], [-0.2, 0.5]]
data2 = np.random.multivariate_normal(mean2, cov2, size=n // 2)

# Combine and Transform to [0, 1]
# The sigmoid function maps (-inf, inf) -> (0, 1) smoothly
data_raw = np.vstack([data1, data2])
data_2d = expit(data_raw) 

# 2. Fit Beta KDE
print("Fitting Beta KDE...")
# We use the default 'beta-reference' rule. 
# It should handle this smooth, bimodal data perfectly.
kde = BetaKDE(bandwidth='beta-reference').fit(data_2d)

# 3. Prepare Grid
grid_pts = np.linspace(0.001, 0.999, 100)
X_grid, Y_grid = np.meshgrid(grid_pts, grid_pts)
XY_flat = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

# 4. Compute Density
Z_log = kde.score_samples(XY_flat)
Z = np.exp(Z_log).reshape(100, 100)

# 5. Plot
plt.figure(figsize=(8, 7))

# Scatter plot of data (subsampled for clarity)
plt.scatter(data_2d[::5, 0], data_2d[::5, 1], s=5, color='gray', label='Data')

# Contour plot of Beta KDE
# We use 'filled' contours for a nicer look
plt.contourf(X_grid, Y_grid, Z, levels=15, cmap='Reds', alpha=0.6)
plt.contour(X_grid, Y_grid, Z, levels=15, colors='darkred', linewidths=0.5)

plt.title("Multivariate Beta KDE (Bimodal Data)", fontsize=14)
plt.xlabel("Variable 1")
plt.ylabel("Variable 2")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.colorbar(label="Density")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.2)

output_path = "assets/2d_copula.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()