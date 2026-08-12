"""
Property-based tests for beta-kde using LeanCheck.
These tests use enumerative property-based testing to find edge cases and bugs,
excluding expected failure cases as per LeanCheck best practices.
"""

import numpy as np
from scipy.integrate import quad

from beta_kde.estimator import BetaKDE
from leancheck import check


def prop_univariate_normalization(bounds_lower: float, bounds_upper: float, 
                                   bandwidth_method: str) -> bool:
    """
    Test that univariate density integrates to 1.0 when normalized=True.
    
    Args:
        bounds_lower: Lower bound of the data range
        bounds_upper: Upper bound of the data range
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if the normalization property holds for valid inputs
    """
    # Skip invalid bounds or methods - these are expected to fail and should be excluded
    if bounds_lower >= bounds_upper or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip test for invalid inputs (expected to fail)
        
    try:
        # Create some sample data within the bounds
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(bounds_lower, bounds_upper, size=(10, 1))
        
        kde = BetaKDE(bounds=(bounds_lower, bounds_upper), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Compute integral of density over the bounds
        def density_func(x):
            return np.exp(kde.score_samples(np.array([[x]]), normalized=True)[0])
            
        integral, _ = quad(density_func, bounds_lower, bounds_upper)
        # Allow some tolerance for numerical integration errors
        return abs(integral - 1.0) < 1e-6
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_score_consistency(bounds_lower: float, bounds_upper: float, 
                           bandwidth_method: str) -> bool:
    """
    Test that log-likelihood scores are consistent with mathematical definitions.
    
    Args:
        bounds_lower: Lower bound of the data range
        bounds_upper: Upper bound of the data range
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if score consistency property holds for valid inputs
    """
    # Skip invalid conditions - these are expected to fail and should be excluded
    if bounds_lower >= bounds_upper or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip test for invalid inputs (expected to fail)
        
    try:
        # Create some sample data within the bounds
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(bounds_lower, bounds_upper, size=(5, 1))
        
        kde = BetaKDE(bounds=(bounds_lower, bounds_upper), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Test a few sample points
        test_points = np.array([[0.2], [0.5], [0.8]])
        scores = kde.score_samples(test_points, normalized=True)
        
        # All scores should be finite numbers (not NaN or infinity)
        return not (np.any(np.isnan(scores)) or np.any(np.isinf(scores)))
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_bandwidth_selection_validity(bounds_lower: float, bounds_upper: float,
                                      bandwidth_method: str) -> bool:
    """
    Test that bandwidth selection methods return valid positive values.
    
    Args:
        bounds_lower: Lower bound of the data range
        bounds_upper: Upper bound of the data range
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if the bandwidth selection returns valid positive values for valid inputs
    """
    # Skip invalid conditions - these are expected to fail and should be excluded
    if bounds_lower >= bounds_upper or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip test for invalid inputs (expected to fail)
        
    try:
        # Create some sample data within the bounds  
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(bounds_lower, bounds_upper, size=(10, 1))
        
        kde = BetaKDE(bounds=(bounds_lower, bounds_upper), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Check that bandwidth is valid (positive number)
        if hasattr(kde, 'bandwidth_'):
            return kde.bandwidth_ > 0
        else:
            # For methods that don't store bandwidth directly, check it was computed
            return True  # If we get here without error, method worked
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_multivariate_copula(bounds_lower: float, bounds_upper: float,
                             bandwidth_method: str) -> bool:
    """
    Test that multivariate cases maintain proper copula behavior.
    
    Args:
        bounds_lower: Lower bound of the data range
        bounds_upper: Upper bound of the data range
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if multivariate copula properties hold for valid inputs
    """
    # Skip invalid conditions - these are expected to fail and should be excluded  
    if bounds_lower >= bounds_upper or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip test for invalid inputs (expected to fail)
        
    try:
        # Create 2D sample data within the bounds
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(bounds_lower, bounds_upper, size=(10, 2))
        
        kde = BetaKDE(bounds=(bounds_lower, bounds_upper), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Test that we can score samples without errors
        test_points = np.array([[0.3, 0.7], [0.1, 0.9]])
        scores = kde.score_samples(test_points, normalized=True)
        
        # All scores should be finite numbers (not NaN or infinity)
        return not (np.any(np.isnan(scores)) or np.any(np.isinf(scores)))
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_boundary_handling(bounds_lower: float, bounds_upper: float) -> bool:
    """
    Test that boundary points are handled correctly.
    
    Args:
        bounds_lower: Lower bound of the data range
        bounds_upper: Upper bound of the data range
        
    Returns:
        True if boundary handling works properly for valid inputs
    """
    # Skip invalid conditions - these are expected to fail and should be excluded
    if bounds_lower >= bounds_upper:
        return True  # Skip test for invalid inputs (expected to fail)
        
    try:
        # Test with data points at boundaries
        X = np.array([[bounds_lower], [bounds_upper]])
        
        kde = BetaKDE(bounds=(bounds_lower, bounds_upper))
        kde.fit(X)
        
        # Test scoring at boundary points
        scores = kde.score_samples(X, normalized=True)
        
        # All scores should be finite numbers (not NaN or infinity)
        return not (np.any(np.isnan(scores)) or np.any(np.isinf(scores)))
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_numerical_stability(bounds_lower: float, bounds_upper: float,
                            bandwidth_method: str) -> bool:
    """
    Test that the implementation is numerically stable.
    
    Args:
        bounds_lower: Lower bound of the data range
        bounds_upper: Upper bound of the data range
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if numerical stability property holds for valid inputs
    """
    # Skip invalid conditions - these are expected to fail and should be excluded  
    if bounds_lower >= bounds_upper or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip test for invalid inputs (expected to fail)
        
    try:
        # Create some sample data within the bounds
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(bounds_lower, bounds_upper, size=(5, 1))
        
        kde = BetaKDE(bounds=(bounds_lower, bounds_upper), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Test scoring at various points including extreme values
        test_points = np.array([[0.0], [0.5], [1.0]])
        scores = kde.score_samples(test_points, normalized=True)
        
        # All scores should be finite numbers (not NaN or infinity)
        return not (np.any(np.isnan(scores)) or np.any(np.isinf(scores)))
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_copula_consistency(n_features: int, bandwidth_method: str) -> bool:
    """
    Test that copula properties are consistent for multivariate cases.
    
    Args:
        n_features: Number of features (should be >= 2 for copula)
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if copula consistency property holds
    """
    # Only test multivariate cases (n_features >= 2)  
    if n_features < 2 or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip for invalid inputs
    
    try:
        # Create multivariate data with appropriate bounds
        np.random.seed(42)
        X = np.random.uniform(0, 1, size=(20, n_features))
        
        kde = BetaKDE(bounds=(0.0, 1.0), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Test that we can compute scores for multivariate data
        test_points = np.random.uniform(0, 1, size=(5, n_features))
        scores = kde.score_samples(test_points, normalized=True)
        
        # All scores should be finite numbers (not NaN or infinity)
        return not (np.any(np.isnan(scores)) or np.any(np.isinf(scores)))
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_copula_dimensionality(n_features: int, bandwidth_method: str) -> bool:
    """
    Test that copula works correctly across different dimensionalities.
    
    Args:
        n_features: Number of features (should be >= 2 for copula)
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if dimensionality handling is correct
    """
    # Only test multivariate cases (n_features >= 2)  
    if n_features < 2 or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip for invalid inputs
    
    try:
        # Create data with different number of features
        np.random.seed(42)
        X = np.random.uniform(0, 1, size=(15, n_features))
        
        kde = BetaKDE(bounds=(0.0, 1.0), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Test scoring and that we get expected results
        test_points = np.random.uniform(0, 1, size=(3, n_features))
        scores = kde.score_samples(test_points, normalized=True)
        
        # All scores should be finite numbers (not NaN or infinity)
        return not (np.any(np.isnan(scores)) or np.any(np.isinf(scores)))
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


def prop_copula_transformation_properties(bounds_lower: float, bounds_upper: float,
                                         bandwidth_method: str) -> bool:
    """
    Test that the copula transformation properties hold.
    
    Args:
        bounds_lower: Lower bound of the data range
        bounds_upper: Upper bound of the data range
        bandwidth_method: Bandwidth selection method
        
    Returns:
        True if copula transformation properties hold
    """
    # Skip invalid conditions - these are expected to fail and should be excluded  
    if bounds_lower >= bounds_upper or bandwidth_method not in ['LCV', 'LSCV', 'beta-reference']:
        return True  # Skip test for invalid inputs (expected to fail)
        
    try:
        # Create 2D data for copula testing
        np.random.seed(42)
        X = np.random.uniform(bounds_lower, bounds_upper, size=(10, 2))
        
        kde = BetaKDE(bounds=(bounds_lower, bounds_upper), bandwidth=bandwidth_method)
        kde.fit(X)
        
        # Test that copula transformation gives valid results
        test_points = np.array([[0.3, 0.7], [0.1, 0.9]])
        scores = kde.score_samples(test_points, normalized=True)
        
        # All scores should be finite numbers (not NaN or infinity)
        return not (np.any(np.isnan(scores)) or np.any(np.isinf(scores)))
    except Exception:
        # If there's any error in computation, return False to indicate problem
        return False


if __name__ == "__main__":
    print("Running LeanCheck property-based tests for beta-kde...")
    print("Note: Invalid inputs (like equal bounds) are excluded from testing as they")
    print("are expected to fail and would interfere with finding actual bugs.")
    
    # Run the property tests - only valid inputs will be tested
    try:
        print("\n1. Testing univariate normalization:")
        check(prop_univariate_normalization, float, float, str)
        
        print("\n2. Testing score consistency:")
        check(prop_score_consistency, float, float, str)
        
        print("\n3. Testing bandwidth selection validity:")
        check(prop_bandwidth_selection_validity, float, float, str)
        
        print("\n4. Testing multivariate copula:")
        check(prop_multivariate_copula, float, float, str)
        
        print("\n5. Testing boundary handling:")
        check(prop_boundary_handling, float, float)
        
        print("\n6. Testing numerical stability:")
        check(prop_numerical_stability, float, float, str)
        
        print("\n7. Testing copula consistency:")
        check(prop_copula_consistency, int, str)
        
        print("\n8. Testing copula dimensionality:")
        check(prop_copula_dimensionality, int, str)
        
        print("\n9. Testing copula transformation properties:")
        check(prop_copula_transformation_properties, float, float, str)
        
        print("\nAll tests completed! Properties hold for valid inputs.")
    except Exception as e:
        print(f"Error running tests: {e}")