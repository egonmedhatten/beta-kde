import time
import numpy as np
import pytest
from beta_kde.estimator import BetaKDE

def test_large_sample_performance():
    """
    Performance test with N=10,000.
    This ensures the code actually runs on large data and doesn't just skip.
    """
    n_samples = 10000
    np.random.seed(42)
    data = np.random.beta(2, 5, size=n_samples)
    
    kde = BetaKDE(bandwidth="MISE_rule")
    
    start_time = time.time()
    kde.fit(data, compute_normalization=True)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nN={n_samples} fit time: {duration:.4f}s")
    
    # Basic assertion: It should be faster than 5 seconds for MISE rule
    # (MISE is O(1), normalization is the slow part)
    assert duration < 5.0
    assert kde.n_samples_ == n_samples