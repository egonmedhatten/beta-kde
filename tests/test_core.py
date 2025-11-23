import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy.testing import assert_allclose
from beta_kde.estimator import BetaKDE
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

# --- Fixtures ---


@pytest.fixture
def simple_data():
    """A simple, well-behaved dataset (Reshaped to 2D column vector)."""
    return np.array([0.2, 0.3, 0.4, 0.5, 0.6]).reshape(-1, 1)


@pytest.fixture
def beta_data():
    """Data that should be valid for rule-of-thumb methods (Reshaped to 2D)."""
    np.random.seed(42)
    return np.random.beta(a=3, b=5, size=100).reshape(-1, 1)


@pytest.fixture
def bad_mise_data():
    """Data that should fail the MISE rule parameter check (Beta(0.1, 0.1))."""
    np.random.seed(42)
    return np.random.beta(a=0.1, b=0.1, size=100).reshape(-1, 1)


# --- Initialization & Parameter Tests ---


def test_init_parameters():
    """Test that parameters are stored correctly in __init__."""
    kde = BetaKDE(
        bandwidth="LCV",
        bounds=(0, 10),
        bandwidth_bounds=(0.05, 0.3),
        integration_points=150,
    )
    assert kde.bandwidth == "LCV"
    assert kde.bounds == (0, 10)
    assert kde.bandwidth_bounds == (0.05, 0.3)
    assert kde.integration_points == 150
    # Should not be fitted yet
    assert not hasattr(kde, "bandwidth_")


@pytest.mark.parametrize("bad_bw", [0.0, -0.1, "invalid_str"])
def test_fit_bad_bandwidth_parameterized(simple_data, bad_bw):
    """Test validation of invalid bandwidth values using parametrization."""
    kde = BetaKDE(bandwidth=bad_bw)
    with pytest.raises(ValueError):
        kde.fit(simple_data)


def test_bad_bounds_init(simple_data):
    """Test that invalid bounds raise an error immediately during fit."""
    # Min > Max
    kde = BetaKDE(bounds=(5, 0))
    with pytest.raises(ValueError, match="strictly increasing"):
        kde.fit(simple_data)

    # Bounds equal
    kde_eq = BetaKDE(bounds=(1, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        kde_eq.fit(simple_data)


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
        # Reshape to 2D
        kde.fit(np.array([-0.1, 0.1, 0.5, 1.2]).reshape(-1, 1))

    # Case 2: Custom bounds (0, 10)
    kde_custom = BetaKDE(bounds=(0, 10))
    # This should pass (Reshaped)
    kde_custom.fit(np.array([2.0, 5.0, 7.0]).reshape(-1, 1))
    # This should fail (Reshaped)
    with pytest.raises(ValueError, match="within the interval"):
        kde_custom.fit(np.array([2.0, 5.0, 7.0, 10.1]).reshape(-1, 1))


def test_input_validation_shapes():
    """Test Scikit-learn style input validation (Strict 2D enforcement)."""
    kde = BetaKDE()

    # 2D Column vector should work (standard sklearn input)
    X_col = np.array([[0.1], [0.2], [0.3]])
    kde.fit(X_col)
    assert kde.n_samples_ == 3

    # 1D array should FAIL now (Strict Sklearn Compliance)
    X_flat = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):  # Expected 2D, got 1D
        kde.fit(X_flat)


# --- Custom Bounds, Scaling & Normalization Tests ---


def test_custom_bounds_scaling():
    """
    Verify that data in [0, 100] works and PDF integrates to ~1.
    """
    # Data in [0, 100]
    np.random.seed(42)
    data = np.random.beta(2, 5, size=100) * 100
    # Must be 2D
    data = data.reshape(-1, 1)

    kde = BetaKDE(bounds=(0, 100), bandwidth="beta-reference")
    kde.fit(data)

    assert kde.is_fitted_
    assert kde.scale_factor_ == 100.0

    # 1. Un-normalized behavior (Asymptotic consistency only)
    # Note: pdf() convenience method handles scalar inputs internally
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
    # This accesses a private method that calculates stats per dimension.
    # It expects a 1D array internally.
    data = np.array([0.4, 0.6, 0.45, 0.55])

    ahat, bhat = kde._estimate_beta_params(data)
    assert_allclose(ahat, 19.5)
    assert_allclose(bhat, 19.5)


def test_estimate_params_zero_variance():
    """Test that zero variance raises an error in parameter estimation."""
    kde = BetaKDE()
    # Internal method expects 1D
    data = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Sample variance is zero"):
        kde._estimate_beta_params(data)


def test_fit_with_exact_zeros_and_ones():
    """
    Ensures the estimator handles exact boundaries by clipping internally.
    """
    dangerous_data = np.array([0.0, 0.1, 0.5, 0.9, 1.0]).reshape(-1, 1)
    kde = BetaKDE(bandwidth=0.1)

    kde.fit(dangerous_data)

    assert kde.is_fitted_
    # Ensure no NaNs in output
    scores = kde.score_samples(dangerous_data)
    assert np.all(np.isfinite(scores))


def test_constant_data_behavior():
    """Test behavior when data has 0 variance (constant)."""
    # 2D Reshape
    data = np.array([0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
    kde = BetaKDE(bandwidth="beta-reference")

    # We allow n>1 constant data to raise error (as per updated code)
    with pytest.raises(ValueError, match="Sample variance is zero"):
        kde.fit(data)


# --- MISE Rule Tests ---


def test_mise_rule_exact_math(beta_data):
    """Test that the ported MISE rule produces the expected result."""
    kde = BetaKDE(bandwidth="beta-reference", verbose=0)
    kde.fit(beta_data)

    assert not kde.is_fallback_
    assert kde.bandwidth_ > 0
    assert kde.bandwidth_ < 1


def test_mise_with_boundaries_sufficient_data():
    """
    Test that MISE works even with 0s and 1s if we have enough data points.
    """
    np.random.seed(42)
    # Generate stable data
    data = np.random.beta(5, 5, size=100)
    # Inject boundaries
    data[0] = 0.0
    data[1] = 1.0

    kde = BetaKDE(bandwidth="beta-reference", verbose=0)
    kde.fit(data.reshape(-1, 1))

    # Should NOT fallback because distribution parameters > 1.5
    assert not kde.is_fallback_
    assert 0 < kde.bandwidth_ < 1


def test_mise_rule_fails_safely(bad_mise_data):
    """Test that MISE rule falls back safely when assumptions are violated."""
    kde = BetaKDE(bandwidth="beta-reference", verbose=1)

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

    # X_test must be 2D
    X_test = np.array([0.25, 0.35]).reshape(-1, 1)

    log_pdf = kde.score_samples(X_test)
    pdf_val = kde.pdf(X_test)

    # Exp(log_pdf) should equal pdf
    assert_allclose(np.exp(log_pdf), pdf_val)


def test_pdf_evaluation_at_boundaries(simple_data):
    """Test behavior at exactly 0.0 and 1.0."""
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(simple_data)

    # 2D input
    eval_pts = np.array([0.0, 1.0]).reshape(-1, 1)
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


# --- Multivariate (Copula) Tests ---


def test_multivariate_integration():
    """
    Critical Test: Verify that a 2D model actually creates a valid
    probability density that integrates to ~1.0.
    """
    # 1. Generate correlated 2D data (e.g., x=y)
    np.random.seed(42)
    n = 200
    x = np.random.beta(2, 2, size=n)
    y = x + np.random.normal(0, 0.1, size=n)

    # Clip to bounds and stack to create (N, 2) array
    data = np.column_stack((np.clip(x, 0.01, 0.99), np.clip(y, 0.01, 0.99)))

    # 2. Fit Model
    kde = BetaKDE(bounds=(0, 1))
    kde.fit(data)

    assert kde.n_features_ == 2
    assert len(kde.marginal_bandwidths_) == 2

    # 3. Integrate PDF over 2D unit square [0,1]x[0,1]
    # Simple Monte Carlo integration
    n_integrate = 5000
    pts = np.random.uniform(0, 1, size=(n_integrate, 2))

    pdf_values = kde.pdf(pts)
    volume = np.mean(pdf_values) * 1.0  # Area is 1x1=1

    # Should be close to 1.0 (allow ~10% error for MC noise)
    assert_allclose(volume, 1.0, rtol=0.1)


def test_multivariate_structure():
    """Check that internal attributes for Copulas are set correctly."""
    data = np.random.rand(50, 3)  # 3 Dimensions
    kde = BetaKDE()
    kde.fit(data)

    # Check Marginals
    assert len(kde.marginal_bandwidths_) == 3
    assert len(kde.x_grids_) == 3
    assert len(kde.cdf_grids_) == 3

    # Check Copula
    assert hasattr(kde, "copula_bandwidth_")
    assert hasattr(kde, "U_train_")
    assert kde.U_train_.shape == data.shape


# --- Sklearn Check ---


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.SkipTestWarning")
def test_sklearn_estimator_check():
    """
    Full check of Scikit-learn estimator compliance.

    We initialize the estimator with wide bounds (-1000, 1000) because
    check_estimator generates random standard normal data (approx -3 to 3).

    This verifies that the API (fit, score_samples, set_params) functions
    correctly, while respecting the strict boundary logic we implemented.
    """
    # Configure a "test-compatible" instance
    est = BetaKDE(bounds=(-1000, 1000))

    # Run the full suite
    check_estimator(est)
