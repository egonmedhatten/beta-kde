import sys
import os

# Add the ../src directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

import numpy as np
import matplotlib.pyplot as plt
from beta_kde import BetaKDE

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

# 1. Generate Correlated Data
np.random.seed(42)
n = 2000
# Simple dependency: y is related to x, both bounded
x = np.random.beta(2, 2, size=n)
y = x + np.random.normal(0, 0.15, size=n)
# Clip to strict [0, 1] bounds
data_2d = np.column_stack((np.clip(x, 0.01, 0.99), np.clip(y, 0.01, 0.99)))

# 2. Fit Beta KDE
kde_2d = BetaKDE(bandwidth='beta-reference').fit(data_2d)

# 3. Evaluate on Grid for Contour Plot
grid_pts = np.linspace(0, 1, 100)
X_grid, Y_grid = np.meshgrid(grid_pts, grid_pts)
XY_flat = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

Z_log = kde_2d.score_samples(XY_flat)
Z = np.exp(Z_log).reshape(100, 100)

# 4. Plot
plt.figure(figsize=(7, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], s=5, alpha=0.3, color='gray', label='Data')
plt.contour(X_grid, Y_grid, Z, levels=15, cmap='Reds', linewidths=1.5)
plt.title("2D Beta Copula Density Estimation")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Density")
plt.legend()
plt.grid(True, alpha=0.3)

# Save
output_path = "assets/2d_copula.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")