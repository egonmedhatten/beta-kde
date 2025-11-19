import sys
import os

# --- Path Setup ---
# Add the project root (one level up) and src/ to sys.path
# This allows the script to find 'beta_kernel' without installation
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from beta_kernel.estimator import BetaKDE
import time

# --- Plotting Style Setup for Academic Papers ---
# Tries to use LaTeX fonts if available, otherwise falls back to robust defaults
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (6, 4), # Standard single-column width
        "lines.linewidth": 1.5,
        "lines.markersize": 6
    })
except:
    # Fallback if LaTeX is not installed
    plt.rcParams.update({
        "font.family": "serif",
        "figure.figsize": (6, 4)
    })

def run_convergence_analysis():
    print("Generating Figure: Integral Convergence vs Sample Size...")
    
    # Sample sizes (log-spaced)
    sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000]
    
    deviations = []
    std_devs = []
    
    # Test Distribution: Beta(2, 5)
    # Moderately skewed, representative of "Nice" data
    a_true, b_true = 2, 5
    n_trials = 30 
    
    for n in sample_sizes:
        trial_errors = []
        print(f"  Processing N={n}...")
        
        for i in range(n_trials):
            np.random.seed(n * 100 + i)
            data = np.random.beta(a_true, b_true, size=n)
            
            # Fit estimator (Unnormalized)
            # using MISE rule for speed
            kde = BetaKDE(bandwidth="MISE_rule", verbose=0)
            kde.fit(data)
            
            # Calculate Integral numerically
            # We expect this to be close to 1.0, but not exactly
            func = lambda x: kde.pdf(x, normalized=False)
            integral, _ = quad(func, 0, 1, limit=50)
            
            # Error = |Integral - 1.0|
            trial_errors.append(abs(integral - 1.0))
            
        deviations.append(np.mean(trial_errors))
        std_devs.append(np.std(trial_errors) / np.sqrt(n_trials)) # Standard Error

    # --- Theoretical Line ---
    # The bandwidth h scales as n^(-0.4) for AMISE optimal rule.
    # The bias in the integral scales roughly with h.
    # So we expect Error ~ O(n^-0.4).
    
    # Anchor the theoretical line to the first data point
    log_n = np.log(sample_sizes)
    
    # Calculate theoretical curve starting from N=50
    theoretical_slope = -0.4
    intercept = np.log(deviations[0]) - (theoretical_slope * np.log(sample_sizes[0]))
    theoretical_y = np.exp(intercept + theoretical_slope * log_n)

    # --- Plotting ---
    fig, ax = plt.subplots()
    
    # Empirical Data
    ax.loglog(sample_sizes, deviations, 'o-', color='#d62728', label='Empirical Deviation', 
              markerfacecolor='white', markeredgewidth=1.5, zorder=10)
    
    # Error bars
    ax.errorbar(sample_sizes, deviations, yerr=std_devs, fmt='none', ecolor='#d62728', alpha=0.5)
    
    # Theoretical Rate
    ax.loglog(sample_sizes, theoretical_y, 'k--', label=r'Theoretical Rate $\mathcal{O}(n^{-0.4})$', alpha=0.7)

    # Aesthetics
    ax.set_xlabel(r'Sample Size ($n$)')
    ax.set_ylabel(r'Mean Absolute Deviation $|\int \hat{f} - 1|$')
    ax.set_title(r'Convergence of Total Probability Mass')
    ax.grid(True, which="major", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    ax.legend()
    
    # Save
    output_file = "integral_convergence.pdf"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    run_convergence_analysis()