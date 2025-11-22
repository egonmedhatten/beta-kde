import sys
import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from beta_kde.estimator import BetaKDE as NewKDE

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
legacy_path = os.path.join(project_root, "paper_validation")

if legacy_path not in sys.path:
    sys.path.insert(0, legacy_path)

try:
    from KDE import BetaKernelKDE as OldKDE
    HAVE_LEGACY = True
except ImportError:
    HAVE_LEGACY = False

@pytest.mark.skipif(not HAVE_LEGACY, reason="Legacy KDE.py not found in paper_validation/")
def test_sanity_check_mise_bandwidth():
    """
    Sanity Check: Ensure the new refactored class produces 
    identically the same bandwidth as the original research script.
    """
    np.random.seed(42)
    raw_data = np.random.beta(3, 5, size=1000)
    
    # Old KDE uses 1D data
    old_kde = OldKDE(bandwidth="MISE_rule", verbose=0)
    old_kde.fit(raw_data)
    bw_old = old_kde.bandwidth
    
    # New KDE requires 2D data
    new_kde = NewKDE(bandwidth="beta-reference", verbose=0)
    new_kde.fit(raw_data.reshape(-1, 1))
    bw_new = new_kde.bandwidth_
    
    assert_allclose(bw_new, bw_old, rtol=1e-7)

@pytest.mark.skipif(not HAVE_LEGACY, reason="Legacy KDE.py not found in paper_validation/")
def test_sanity_check_pdf_values():
    """
    Sanity Check: Ensure PDF evaluations are identical.
    Using normalized=False to match original script behavior.
    """
    np.random.seed(42)
    raw_data = np.random.beta(3, 5, size=100)
    eval_points = np.linspace(0.01, 0.99, 10)
    
    # Old KDE
    old_kde = OldKDE(bandwidth=0.1, verbose=0)
    old_kde.fit(raw_data)
    pdf_old = old_kde.pdf(eval_points)
    
    # New KDE (Fit needs 2D)
    new_kde = NewKDE(bandwidth=0.1, verbose=0)
    new_kde.fit(raw_data.reshape(-1, 1))
    
    # The pdf() convenience method we added SHOULD handle 1D eval points,
    # but strictly speaking, passing 2D is safer. 
    # Let's pass 2D for consistency with the strict API.
    pdf_new = new_kde.pdf(eval_points.reshape(-1, 1), normalized=False)
    
    # Output of pdf() is 1D array in both cases
    assert_allclose(pdf_new, pdf_old, rtol=1e-7)