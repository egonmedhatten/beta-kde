import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from src.beta_kernel.estimator import BetaKDE
import matplotlib.pyplot as plt

def true_pdf(x, a, b):
    return beta.pdf(x, a, b)

def compute_ise(kde, true_a, true_b, normalized=False):
    """Computes the Integrated Squared Error (ISE) against the true Beta PDF."""
    
    # Function to integrate: (f_hat(x) - f_true(x))^2
    def squared_error(x):
        # Get estimated density
        # We use the scalar version of pdf() for quad integration
        est = kde.pdf(x, normalized=normalized)
        true = true_pdf(x, true_a, true_b)
        return (est - true)**2

    # Integrate over support [0, 1]
    ise, _ = quad(squared_error, 0, 1, limit=100)
    return ise

def run_comparison(n_samples=100, n_trials=50):
    print(f"Running comparison: N={n_samples}, Trials={n_trials}")
    print("-" * 60)
    
    # Parameters for true distribution (Beta(2, 12) - skewed but 'nice')
    a, b = 2, 12
    
    ise_unnorm = []
    ise_norm = []
    
    for i in range(n_trials):
        np.random.seed(42 + i) # Reproducibility
        data = np.random.beta(a, b, size=n_samples)
        
        # Fit KDE using MISE rule (fast)
        # We compute normalization constant once during fit to speed up the normalized pdf calls
        kde = BetaKDE(bandwidth="MISE_rule", verbose=0)
        kde.fit(data, compute_normalization=True) 
        
        # Compute ISE for both variants
        err_u = compute_ise(kde, a, b, normalized=False)
        err_n = compute_ise(kde, a, b, normalized=True)
        
        ise_unnorm.append(err_u)
        ise_norm.append(err_n)
        
        if (i+1) % 10 == 0:
            print(f"Trial {i+1}/{n_trials} completed.")

    mean_u = np.mean(ise_unnorm)
    mean_n = np.mean(ise_norm)
    
    # Percent difference relative to unnormalized error
    diff_percent = ((mean_u - mean_n) / mean_u) * 100
    
    print("-" * 60)
    print(f"Mean ISE (Unnormalized): {mean_u:.6f}")
    print(f"Mean ISE (Normalized):   {mean_n:.6f}")
    print(f"Difference:              {mean_u - mean_n:.6f}")
    print(f"Relative Improvement:    {diff_percent:.2f}%")
    print("-" * 60)
    
    return mean_u, mean_n

if __name__ == "__main__":
    # Run a small experiment
    run_comparison(n_samples=1000, n_trials=500)