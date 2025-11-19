import sys
import os
import numpy as np
import pytest
from numpy.testing import assert_allclose

# --- Path Setup ---
# Add the project root to sys.path so we can import the original 'KDE.py'
# This assumes the tests are running from the 'tests/' directory or root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
from beta_kernel.estimator import BetaKDE as NewKDE

# Try importing the legacy script. If it's missing, skip these tests.
try:
    from paper_validation.KDE import BetaKernelKDE as OldKDE
    HAVE_LEGACY = True
except ImportError:
    HAVE_LEGACY = False

@pytest.mark.skipif(not HAVE_LEGACY, reason="Legacy KDE.py not found in project root")
def test_sanity_check_mise_bandwidth():
    """
    Sanity Check: Ensure the new refactored class produces 
    identically the same bandwidth as the original research script
    for a fixed random seed.
    """
    np.random.seed(42)
    # Generate standard Beta data
    data = np.random.beta(3, 5, size=10000)
    
    # --- Run Old Implementation ---
    old_kde = OldKDE(bandwidth="MISE_rule", verbose=0)
    old_kde.fit(data)
    bw_old = old_kde.bandwidth
    
    # --- Run New Implementation ---
    new_kde = NewKDE(bandwidth="MISE_rule", verbose=0)
    new_kde.fit(data)
    bw_new = new_kde.bandwidth_
    
    print(f"\nOld Bandwidth: {bw_old:.8f}")
    print(f"New Bandwidth: {bw_new:.8f}")
    
    # Assert identical to 6 decimal places (strict regression test)
    assert_allclose(bw_new, bw_old, rtol=1e-7)

@pytest.mark.skipif(not HAVE_LEGACY, reason="Legacy KDE.py not found in project root")
def test_sanity_check_pdf_values():
    """
    Sanity Check: Ensure PDF evaluations are identical.
    We must use normalized=False in the new package to match the 
    asymptotic (unnormalized) behavior of the old script.
    """
    np.random.seed(42)
    data = np.random.beta(3, 5, size=10000)
    # Evaluation points strictly inside [0, 1] to avoid boundary handling differences
    # (The new script handles exact 0/1 slightly more gracefully/safely than the old one)
    eval_points = np.linspace(0.01, 0.99, 10)
    
    # --- Old ---
    old_kde = OldKDE(bandwidth=0.1, verbose=0)
    old_kde.fit(data)
    pdf_old = old_kde.pdf(eval_points)
    
    # --- New ---
    new_kde = NewKDE(bandwidth=0.1, verbose=0)
    new_kde.fit(data)
    
    # IMPORTANT: The old script is NOT normalized. 
    # The new script defaults to normalized=False.
    # So they should match exactly.
    pdf_new = new_kde.pdf(eval_points, normalized=False)
    
    assert_allclose(pdf_new, pdf_old, rtol=1e-7)