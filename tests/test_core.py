import os

# Import BetaKDEClassifier from examples
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from beta_kde.estimator import BetaKDE

examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
sys.path.insert(0, examples_path)
from generative_classifier import BetaKDEClassifier

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
        fig, _ax = kde.plot(show_histogram=True)
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
    
    Since BetaKDE is designed for bounded data (default [0,1]), we use wide bounds
    (-1000, 1000) to allow sklearn's check_estimator to generate random normal data
    without constant failures. The key API checks still validate compatibility.
    
    For proper bounded data testing, see test_sklearn_with_bounded_data below.
    """
    # Configure a "test-compatible" instance with wide bounds
    est = BetaKDE(bounds=(-1000, 1000))
    
    # Mark checks that are expected to fail due to bounded constraints
    # These checks generate data that violates our bounds by design
    expected_failed_checks = {
        "check_fit2d_1feature": "1D data generation may violate bounds",
        "check_fit1d": "1D validation handled separately",
        "check_fit2d_predict1d": "Predictions depend on bounds",
    }
    
    # Run the full suite with expected failures
    check_estimator(est, expected_failed_checks=expected_failed_checks)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.SkipTestWarning")
def test_sklearn_with_bounded_data():
    """
    Test BetaKDE compliance with bounded data constraints.
    
    This test validates that the estimator properly handles data within bounds
    and rejects data outside bounds, which is the core functionality of BetaKDE.
    
    We use wide bounds (-1000, 1000) to allow sklearn's check_estimator to generate
    synthetic data, then separately validate bounded behavior with appropriate tests.
    """
    # Use wide bounds to allow sklearn checks to run
    est = BetaKDE(bounds=(-1000, 1000), bandwidth=0.1)
    
    # Mark checks that don't make sense for density estimators
    expected_failed_checks = {
        "check_fit2d_1feature": "1D data generation may violate bounds",
        "check_fit1d": "1D validation handled separately",
        "check_fit2d_predict1d": "Predictions depend on bounds",
        "check_dtype_object": "Object dtype not relevant for density estimation",
        "check_complex_data": "Complex data not supported",
        "check_positive_only": "Wide bounds allow negative values",
    }
    
    # Run checks with bounded instance
    check_estimator(est, expected_failed_checks=expected_failed_checks)
    
    # Additional validation for bounded behavior with narrow bounds
    est_narrow = BetaKDE(bounds=(0, 1), bandwidth=0.1)
    
    # Generate data that should work (within bounds)
    np.random.seed(42)
    bounded_data = np.random.beta(2, 5, size=50).reshape(-1, 1)
    est_narrow.fit(bounded_data)
    assert est_narrow.is_fitted_
    
    # Verify PDF integrates to ~1 over bounded region
    # Sample-based integration since pdf expects 2D input
    integration_points = np.linspace(0, 1, 100).reshape(-1, 1)
    pdf_values = est_narrow.pdf(integration_points)
    # Use scipy's trapezoid (numpy.trapz was removed in NumPy 2.0)
    from scipy.integrate import trapezoid
    integral = trapezoid(pdf_values.flatten(), integration_points.flatten())
    assert_allclose(integral, 1.0, rtol=0.1)
    
    # Generate data outside bounds (should fail)
    out_of_bounds_data = np.array([[-0.1], [0.2], [0.5], [1.2]])
    est_out = BetaKDE(bounds=(0, 1), bandwidth=0.1)
    with pytest.raises(ValueError, match="within the interval"):
        est_out.fit(out_of_bounds_data)
    
    # Test predictions at boundaries (should be finite)
    boundary_points = np.array([[0.0], [0.5], [1.0]])
    scores = est_narrow.score_samples(boundary_points)
    assert np.all(np.isfinite(scores))


# --- Edge Case Tests ---


def test_edge_case_all_zeros():
    """Test behavior with data all at 0.0."""
    data = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
    kde = BetaKDE(bandwidth=0.05)
    
    # Should fit with fallback
    kde.fit(data)
    assert kde.is_fitted_
    
    # Should return finite scores
    scores = kde.score_samples(data)
    assert np.all(np.isfinite(scores))


def test_edge_case_all_ones():
    """Test behavior with data all at 1.0."""
    data = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]])
    kde = BetaKDE(bandwidth=0.05)
    
    # Should fit with fallback
    kde.fit(data)
    assert kde.is_fitted_
    
    # Should return finite scores
    scores = kde.score_samples(data)
    assert np.all(np.isfinite(scores))


def test_edge_case_mixed_boundaries():
    """Test behavior with data at both boundaries."""
    data = np.array([[0.0], [0.0], [1.0], [1.0], [0.5]])
    kde = BetaKDE(bandwidth=0.05)
    
    # Should fit with fallback
    kde.fit(data)
    assert kde.is_fitted_
    
    # Should return finite scores
    scores = kde.score_samples(data)
    assert np.all(np.isfinite(scores))


def test_edge_case_constant_data_normalized():
    """Test that normalized scoring fails gracefully for constant data."""
    data = np.array([[0.5], [0.5], [0.5]])
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(data)
    
    # Unnormalized should work
    scores_unnorm = kde.score_samples(data, normalized=False)
    assert np.all(np.isfinite(scores_unnorm))
    
    # Normalized should fail with informative error
    with pytest.raises(ValueError, match="constant data"):
        kde.score_samples(data, normalized=True)


def test_edge_case_3d_multivariate():
    """Test 3D multivariate estimation."""
    np.random.seed(42)
    n = 100
    data = np.random.rand(n, 3)
    
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(data)
    
    assert kde.n_features_ == 3
    assert len(kde.marginal_bandwidths_) == 3
    
    # Should be able to score
    test_points = np.random.rand(10, 3)
    scores = kde.score_samples(test_points)
    assert len(scores) == 10
    assert np.all(np.isfinite(scores))


def test_edge_case_4d_multivariate():
    """Test 4D multivariate estimation."""
    np.random.seed(42)
    n = 150
    data = np.random.rand(n, 4)
    
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(data)
    
    assert kde.n_features_ == 4
    
    # Should be able to score
    test_points = np.random.rand(5, 4)
    scores = kde.score_samples(test_points)
    assert len(scores) == 5


def test_normalization_constant_edge_cases():
    """Test normalization constant error handling."""
    # Test with too few samples
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(np.array([[0.5]]))  # Only 1 sample
    
    with pytest.raises(ValueError, match="At least 2 samples"):
        _ = kde.normalization_constant
    
    # Test with constant data in 2D
    kde2 = BetaKDE(bandwidth=0.05)
    kde2.fit(np.array([[0.5, 0.3], [0.5, 0.3], [0.5, 0.3]]))
    
    with pytest.raises(ValueError, match="constant data"):
        _ = kde2.normalization_constant


def test_score_consistency_with_normalization():
    """Test that score() uses normalized=True."""
    np.random.seed(42)
    data = np.random.beta(2, 5, size=50).reshape(-1, 1)
    
    kde = BetaKDE(bandwidth=0.1)
    kde.fit(data)
    
    # score() should equal sum of normalized score_samples()
    score_method = kde.score(data)
    score_sum = kde.score_samples(data, normalized=True).sum()
    
    assert_allclose(score_method, score_sum)


def test_generative_classifier_workflow():
    """Integration test: Generative classification with BetaKDEClassifier."""
    np.random.seed(42)
    
    # Create synthetic bounded data (two classes with different distributions)
    n_samples = 200
    
    # Class 0: Beta(2, 5) - concentrated near 0
    X_class0 = np.random.beta(2, 5, size=(n_samples, 2))
    y_class0 = np.zeros(n_samples)
    
    # Class 1: Beta(5, 2) - concentrated near 1
    X_class1 = np.random.beta(5, 2, size=(n_samples, 2))
    y_class1 = np.ones(n_samples)
    
    # Combine data
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit BetaKDEClassifier
    clf = BetaKDEClassifier(bandwidth='beta-reference', bounds=(0, 1))
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Check that accuracy is significantly better than chance (50%)
    # With well-separated distributions, we expect >80% accuracy
    accuracy = np.mean(y_pred == y_test)
    assert accuracy > 0.75, f"Expected accuracy > 75%, got {accuracy:.1%}"
    
    # Check that predict_log_proba returns correct shape
    log_proba = clf.predict_log_proba(X_test)
    assert log_proba.shape == (len(X_test), 2), "Log probability shape incorrect"
    
    # Check that probabilities sum to 1 (numerical stability check)
    probs = np.exp(log_proba)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities should sum to 1"


def test_multivariate_correlated_data():
    """Integration test: Multivariate Beta copula with correlated real-world data."""
    np.random.seed(42)
    
    # Generate correlated data (simulating real-world multi-feature data)
    n_samples = 300
    
    # Create correlation structure (like feature correlations in real data)
    mean = [0.5, 0.5, 0.5]
    # Covariance matrix with correlations
    cov = [[0.02, 0.015, 0.01],
           [0.015, 0.03, 0.02],
           [0.01, 0.02, 0.025]]
    
    # Generate multivariate normal, then transform to uniform via CDF
    data_normal = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    # Clip to [0, 1] bounds (simulating bounded real-world measurements)
    data_uniform = np.clip(data_normal, 0, 1)
    
    # Fit multivariate BetaKDE
    kde = BetaKDE(bandwidth='beta-reference', bounds=(0, 1))
    kde.fit(data_uniform)
    
    # Test density evaluation
    test_points = np.random.uniform(0, 1, size=(50, 3))
    log_densities = kde.score_samples(test_points, normalized=True)
    
    # Densities should be finite
    assert np.all(np.isfinite(log_densities)), "Log densities should be finite"
    
    # Test PDF evaluation
    pdf_values = kde.pdf(test_points, normalized=True)
    assert np.all(pdf_values > 0), "PDF values should be positive"
    assert np.all(np.isfinite(pdf_values)), "PDF values should be finite"
    
    # Test that density integrates to ~1 (rough check)
    # Use Monte Carlo integration with uniform sampling
    # ∫p(x)dx ≈ (1/N) * Σp(x_i) where x_i ~ Uniform(0,1)
    n_mc = 1000
    mc_samples = np.random.uniform(0, 1, size=(n_mc, 3))
    mc_densities = kde.pdf(mc_samples, normalized=True)
    integral_estimate = np.mean(mc_densities)  # Since uniform has density 1 over [0,1]^3
    
    # For a properly normalized density, integral ≈ 1
    assert 0.5 < integral_estimate < 2.0, f"Integral estimate should be around 1, got {integral_estimate:.2f}"


def test_cross_validation_workflow():
    """Integration test: Cross-validation with BetaKDE using GridSearchCV."""
    from sklearn.model_selection import GridSearchCV
    
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 150
    X = np.random.beta(3, 4, size=(n_samples, 1))
    
    # Define parameter grid
    param_grid = {
        'bandwidth': ['beta-reference', 0.05, 0.1, 0.2]
    }
    
    # Setup GridSearchCV with BetaKDE
    kde = BetaKDE(bounds=(0, 1))
    grid_search = GridSearchCV(
        kde, param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X)
    
    # Check that best parameters are found
    assert hasattr(grid_search, 'best_params_'), "GridSearchCV should have best_params_"
    assert 'bandwidth' in grid_search.best_params_, "Best params should include bandwidth"
    
    # Check that best estimator has been fitted (note: sklearn uses is_fitted_)
    assert hasattr(grid_search.best_estimator_, 'is_fitted_'), "Best estimator should be fitted"
    assert grid_search.best_estimator_.is_fitted_, "Best estimator should be fitted"
    
    # Test prediction with best model
    best_kde = grid_search.best_estimator_
    test_points = np.linspace(0, 1, 20).reshape(-1, 1)
    
    log_densities = best_kde.score_samples(test_points)
    assert len(log_densities) == len(test_points), "Should evaluate at all test points"
    assert np.all(np.isfinite(log_densities)), "Log densities should be finite"


def test_boundary_bias_reduction():
    """Integration test: Demonstrate boundary bias reduction with BetaKDE."""
    np.random.seed(42)
    
    # Create data clustered near boundaries (where Gaussian KDE would fail)
    n_samples = 200
    
    # Bimodal data with peaks near 0 and 1
    n_peak0 = int(n_samples * 0.6)
    n_peak1 = n_samples - n_peak0
    
    data_peak0 = np.random.beta(2, 8, size=(n_peak0, 1))  # Peak near 0
    data_peak1 = np.random.beta(8, 2, size=(n_peak1, 1))  # Peak near 1
    
    X = np.vstack([data_peak0, data_peak1])
    
    # Fit BetaKDE
    kde_beta = BetaKDE(bandwidth='beta-reference', bounds=(0, 1))
    kde_beta.fit(X)
    
    # For comparison, fit Gaussian KDE (sklearn)
    from sklearn.neighbors import KernelDensity
    kde_gaussian = KernelDensity(bandwidth=0.1)
    kde_gaussian.fit(X)
    
    # Evaluate on fine grid
    grid = np.linspace(0, 1, 100).reshape(-1, 1)
    
    beta_log_dens = kde_beta.score_samples(grid)
    gaussian_log_dens = kde_gaussian.score_samples(grid)
    
    # BetaKDE should give reasonable density near boundaries
    # (Gaussian KDE often underestimates near boundaries due to bias)
    beta_dens = np.exp(beta_log_dens)
    np.exp(gaussian_log_dens)
    
    # BetaKDE densities should be positive everywhere (no negative values)
    assert np.all(beta_dens > 0), "BetaKDE densities should be positive"
    
    # Check that BetaKDE doesn't have artificial dip in the middle
    # (Gaussian KDE often creates artificial valleys near boundaries)
    mid_idx = len(grid) // 2
    left_avg = np.mean(beta_dens[:mid_idx])
    right_avg = np.mean(beta_dens[mid_idx:])
    
    # The ratio should not be extremely small (no artificial gap)
    ratio = min(left_avg, right_avg) / max(left_avg, right_avg)
    assert ratio > 0.1, "BetaKDE should not create artificial gaps in density"
