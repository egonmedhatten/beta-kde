import numpy as np
import pytest
import matplotlib.pyplot as plt
import warnings
from scipy.integrate import quad
from numpy.testing import assert_allclose
from beta_kernel.estimator import BetaKDE
from sklearn.exceptions import NotFittedError, SkipTestWarning
from sklearn.utils.estimator_checks import check_estimator

@pytest.fixture
def simple_data():
    """A simple, well-behaved dataset."""
    return np.array([0.2, 0.3, 0.4, 0.5, 0.6])

@pytest.fixture
def beta_data():
    """Data that should be valid for rule-of-thumb methods."""
    np.random.seed(42)
    return np.random.beta(a=3, b=5, size=100)

@pytest.fixture
def bad_mise_data():
    """Data that should fail the MISE rule parameter check (Beta(0.1, 0.1))."""
    np.random.seed(42)
    return np.random.beta(a=0.1, b=0.1, size=100)

# --- Initialization & Parameter Tests ---

def test_init_parameters():
    """Test that parameters are stored correctly in __init__."""
    kde = BetaKDE(
        bandwidth="LCV", 
        bounds=(0, 10),
        bandwidth_bounds=(0.05, 0.3),
        integration_points=150
    )
    assert kde.bandwidth == "LCV"
    assert kde.bounds == (0, 10)
    assert kde.bandwidth_bounds == (0.05, 0.3)
    assert kde.integration_points == 150
    # Should not be fitted yet
    assert not hasattr(kde, "bandwidth_")

def test_fit_bad_bandwidth_value(simple_data):
    """Test validation of invalid bandwidth values during fit."""
    # Zero bandwidth
    kde = BetaKDE(bandwidth=0.0)
    with pytest.raises(ValueError, match="Bandwidth must be positive"):
        kde.fit(simple_data)

    # Negative bandwidth
    kde = BetaKDE(bandwidth=-0.1)
    with pytest.raises(ValueError, match="Bandwidth must be positive"):
        kde.fit(simple_data)

    # Bad string method
    kde = BetaKDE(bandwidth="invalid_method")
    with pytest.raises(ValueError, match="Unknown bandwidth selection method"):
        kde.fit(simple_data)

def test_fit_ignores_y(simple_data):
    """Test that passing 'y' does not break fit (Sklearn API standard)."""
    kde = BetaKDE(bandwidth=0.1)
    # y can be anything, it should be ignored
    kde.fit(simple_data, y=np.ones(len(simple_data)))
    assert kde.is_fitted_

# --- Data Validation Tests ---

def test_validate_data_range(simple_data):
    """Test that data outside bounds raises ValueError."""
    # Case 1: Default bounds (0, 1)
    kde = BetaKDE()
    with pytest.raises(ValueError, match="within the interval"):
        kde.fit(np.array([-0.1]))
        
    # Case 2: Custom bounds (0, 10)
    kde_custom = BetaKDE(bounds=(0, 10))
    # This should pass
    kde_custom.fit(np.array([5.0]))
    # This should fail
    with pytest.raises(ValueError, match="within the interval"):
        kde_custom.fit(np.array([10.1]))

def test_input_validation_shapes():
    """Test Scikit-learn style input validation."""
    kde = BetaKDE()
    
    # 2D Column vector should work (standard sklearn input)
    X_col = np.array([[0.1], [0.2], [0.3]])
    kde.fit(X_col)
    assert kde.n_samples_ == 3
    
    # 2D Wide array should fail (as per logic)
    X_wide = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="Data must be 1D or a single column"):
        kde.fit(X_wide)

# --- Custom Bounds, Scaling & Normalization Tests ---

def test_custom_bounds_scaling():
    """
    Verify that data in [0, 100] works and PDF integrates to ~1.
    Now also tests the explicit normalization feature.
    """
    # Data in [0, 100]
    np.random.seed(42)
    data = np.random.beta(2, 5, size=100) * 100
    
    kde = BetaKDE(bounds=(0, 100), bandwidth="MISE_rule")
    kde.fit(data)
    
    assert kde.is_fitted_
    assert kde.scale_factor_ == 100.0
    
    # 1. Un-normalized behavior (Asymptotic consistency only)
    func_unnorm = lambda x: kde.pdf(x, normalized=False)
    integral_1, _ = quad(func_unnorm, 0, 100)
    assert_allclose(integral_1, 1.0, rtol=2e-2)
    
    # 2. Normalized behavior (Should be exactly 1.0)
    func_norm = lambda x: kde.pdf(x, normalized=True)
    integral_2, _ = quad(func_norm, 0, 100)
    assert_allclose(integral_2, 1.0, rtol=1e-5)

def test_normalization_caching(simple_data):
    """
    Verify that compute_normalization=True in fit() pre-calculates
    and caches the constant.
    """
    # Case 1: Default (Lazy loading)
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(simple_data)
    assert kde.normalization_constant_ is None
    
    # Call PDF with normalization -> triggers computation + caching
    _ = kde.pdf(0.5, normalized=True)
    assert kde.normalization_constant_ is not None
    
    # Case 2: Pre-computed in fit
    kde_pre = BetaKDE(bandwidth=0.1)
    kde_pre.fit(simple_data, compute_normalization=True)
    assert kde_pre.normalization_constant_ is not None

# --- Logic & Calculation Tests ---

def test_estimate_params_logic():
    """Test the internal method of moments estimation."""
    kde = BetaKDE()
    # Data from original test_estimate_params_doctest_example
    data = np.array([0.4, 0.6, 0.45, 0.55])
    
    # Access private method for unit testing logic
    ahat, bhat = kde._estimate_beta_params(data)
    assert_allclose(ahat, 19.5)
    assert_allclose(bhat, 19.5)

def test_estimate_params_zero_variance():
    """Test that zero variance raises an error in parameter estimation."""
    kde = BetaKDE()
    data = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Sample variance is zero"):
        kde._estimate_beta_params(data)

def test_fit_with_exact_zeros_and_ones():
    """
    Ensures the estimator handles exact boundaries by clipping internally.
    """
    dangerous_data = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    kde = BetaKDE(bandwidth=0.1)
    
    kde.fit(dangerous_data)
        
    assert kde.is_fitted_
    # Ensure no NaNs in output
    scores = kde.score_samples(dangerous_data)
    assert np.all(np.isfinite(scores))

def test_constant_data_behavior():
    """Test behavior when data has 0 variance (constant)."""
    # This tests the fallback logic in _select_bandwidth_mise_rule
    # when variance is 0.
    data = np.array([0.5, 0.5, 0.5, 0.5])
    kde = BetaKDE(bandwidth="MISE_rule")
    
    kde.fit(data)
    assert kde.bandwidth_ == 1e-5
    assert kde.is_fallback_

# --- MISE Rule Tests ---

def test_mise_rule_exact_math(beta_data):
    """Test that the ported MISE rule produces the expected result."""
    kde = BetaKDE(bandwidth="MISE_rule", verbose=0)
    kde.fit(beta_data)
    
    assert not kde.is_fallback_
    assert kde.bandwidth_ > 0
    assert kde.bandwidth_ < 1

def test_mise_with_boundaries_sufficient_data():
    """
    Test that MISE works even with 0s and 1s if we have enough data points
    to keep the variance reasonable.
    """
    np.random.seed(42)
    # Generate stable data
    data = np.random.beta(5, 5, size=100)
    # Inject boundaries
    data[0] = 0.0
    data[1] = 1.0
    
    kde = BetaKDE(bandwidth="MISE_rule", verbose=0)
    kde.fit(data)
    
    # Should NOT fallback because distribution parameters > 1.5
    assert not kde.is_fallback_
    assert 0 < kde.bandwidth_ < 1

def test_mise_rule_fails_safely(bad_mise_data):
    """Test that MISE rule falls back safely when assumptions are violated."""
    kde = BetaKDE(bandwidth="MISE_rule", verbose=1)
    
    # Should warn about fallback
    with pytest.warns(RuntimeWarning, match="MISE Rule failed"):
        kde.fit(bad_mise_data)
        
    assert kde.is_fallback_
    assert kde.bandwidth_ > 0

# --- LCV / LSCV Tests ---

def test_lcv_selection(simple_data):
    """Test LCV bandwidth selection."""
    kde = BetaKDE(bandwidth="LCV", bandwidth_bounds=(0.01, 0.5), verbose=0)
    kde.fit(simple_data)
    assert 0.01 <= kde.bandwidth_ <= 0.5

def test_lscv_selection(simple_data):
    """Test LSCV bandwidth selection."""
    kde = BetaKDE(bandwidth="LSCV", bandwidth_bounds=(0.01, 0.5), verbose=0)
    kde.fit(simple_data)
    assert 0.01 <= kde.bandwidth_ <= 0.5

def test_lscv_custom_grid(simple_data):
    """Test LSCV with custom grid points."""
    kde = BetaKDE(bandwidth="LSCV", selection_grid_points=5, verbose=0)
    kde.fit(simple_data)
    assert kde.is_fitted_

# --- API & Workflow Tests ---

def test_fit_and_attributes(simple_data):
    """Test that fit populates attributes correctly."""
    kde = BetaKDE(bandwidth=0.15)
    kde.fit(simple_data)
    
    assert hasattr(kde, "is_fitted_")
    assert hasattr(kde, "n_samples_")
    assert kde.n_samples_ == 5
    assert kde.bandwidth_ == 0.15
    assert not kde.is_fallback_

def test_score_samples_not_fitted(simple_data):
    """Test that calling score_samples before fit raises error."""
    kde = BetaKDE(bandwidth=0.1)
    with pytest.raises(NotFittedError):
        kde.score_samples(simple_data)

def test_score_samples_consistency(simple_data):
    """Test that score_samples returns log(pdf)."""
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(simple_data)
    
    X_test = np.array([0.25, 0.35])
    log_pdf = kde.score_samples(X_test)
    pdf_val = kde.pdf(X_test)
    
    # Exp(log_pdf) should equal pdf
    assert_allclose(np.exp(log_pdf), pdf_val)

def test_pdf_evaluation_at_boundaries(simple_data):
    """Test behavior at exactly 0.0 and 1.0."""
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(simple_data)
    
    eval_pts = np.array([0.0, 1.0])
    pdf_vals = kde.pdf(eval_pts)
    
    assert np.all(np.isfinite(pdf_vals))
    assert np.all(pdf_vals >= 0)

def test_plot_method(simple_data):
    """Test that the plot method runs without error."""
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(simple_data)
    
    # Smoke test for plotting
    try:
        fig, ax = kde.plot(show_histogram=True)
        plt.close(fig) 
    except Exception as e:
        pytest.fail(f"Plotting failed: {e}")

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.SkipTestWarning")
def test_sklearn_estimator_check():
    """
    Full check of Scikit-learn estimator compliance.
    
    We ignore SkipTestWarning because standard sklearn checks don't 
    currently support a data generator for tags combination:
    {one_d_array=True, positive_only=True}. 
    This is a limitation of check_estimator, not the class.
    """
    check_estimator(BetaKDE())