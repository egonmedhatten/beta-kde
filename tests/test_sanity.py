import sys
import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from beta_kernel.estimator import BetaKDE as NewKDE

# --- Path Setup ---
# We need to import the legacy 'KDE.py' from the 'paper_validation' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
legacy_path = os.path.join(project_root, "paper_validation")

# Add the legacy folder to sys.path so we can import 'KDE'
if legacy_path not in sys.path:
    sys.path.insert(0, legacy_path)

# --- Imports ---
try:
    from KDE import BetaKernelKDE as OldKDE
    HAVE_LEGACY = True
except ImportError:
    # This handles the case where the folder is missing (e.g. inside the pip installed version)
    HAVE_LEGACY = False

@pytest.mark.skipif(not HAVE_LEGACY, reason="Legacy KDE.py not found in paper_validation/")
def test_sanity_check_mise_bandwidth():
    """
    Sanity Check: Ensure the new refactored class produces 
    identically the same bandwidth as the original research script.
    """
    np.random.seed(42)
    data = np.random.beta(3, 5, size=100)
    
    old_kde = OldKDE(bandwidth="MISE_rule", verbose=0)
    old_kde.fit(data)
    bw_old = old_kde.bandwidth
    
    new_kde = NewKDE(bandwidth="MISE_rule", verbose=0)
    new_kde.fit(data)
    bw_new = new_kde.bandwidth_
    
    assert_allclose(bw_new, bw_old, rtol=1e-7)

@pytest.mark.skipif(not HAVE_LEGACY, reason="Legacy KDE.py not found in paper_validation/")
def test_sanity_check_pdf_values():
    """
    Sanity Check: Ensure PDF evaluations are identical.
    Using normalized=False to match original script behavior.
    """
    np.random.seed(42)
    data = np.random.beta(3, 5, size=100)
    eval_points = np.linspace(0.01, 0.99, 10)
    
    old_kde = OldKDE(bandwidth=0.1, verbose=0)
    old_kde.fit(data)
    pdf_old = old_kde.pdf(eval_points)
    
    new_kde = NewKDE(bandwidth=0.1, verbose=0)
    new_kde.fit(data)
    
    pdf_new = new_kde.pdf(eval_points, normalized=False)
    
    assert_allclose(pdf_new, pdf_old, rtol=1e-7)