import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special as sp
from scipy.stats import beta as beta_dist
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_array, check_is_fitted


class BetaKDE(BaseEstimator, DensityMixin):
    """
    Beta Kernel Density Estimation with Scikit-learn API compatibility.
    Supports any bounded interval [a, b] via linear transformation.
    
    Parameters
    ----------
    bandwidth : float, str, or None, default=None
        The bandwidth of the kernel (defined on the scaled [0, 1] space).
        - If float: It is the fixed bandwidth.
        - If str: Must be one of ['LCV', 'LSCV', 'MISE_rule'].
        - If None: Defaults to 'MISE_rule'.
        
    bounds : tuple of float, default=(0.0, 1.0)
        The support of the data (min, max). Data outside this range during fit
        will raise an error.
        
    bandwidth_bounds : tuple of float, default=(0.01, 0.2)
        The lower and upper bounds for bandwidth selection optimization (LCV/LSCV).
        
    selection_grid_points : int, default=30
        Number of grid points for LSCV grid search stage.
        
    heuristic_factor : float, default=4.0
        Factor for heuristic range determination in LSCV.
        
    integration_points : int, default=200
        Number of integration points used for the LSCV objective function.
        
    verbose : int, default=1
        Verbosity level.
    """

    VALID_SELECTION_METHODS = ["LCV", "LSCV", "MISE_rule"]

    def __init__(
        self,
        bandwidth: Optional[Union[float, str]] = None,
        bounds: Tuple[float, float] = (0.0, 1.0),
        bandwidth_bounds: Tuple[float, float] = (0.01, 0.2),
        selection_grid_points: int = 30,
        heuristic_factor: float = 4.0,
        integration_points: int = 200,
        verbose: int = 1,
    ):
        self.bandwidth = bandwidth
        self.bounds = bounds
        self.bandwidth_bounds = bandwidth_bounds
        self.selection_grid_points = selection_grid_points
        self.heuristic_factor = heuristic_factor
        self.integration_points = integration_points
        self.verbose = verbose

    def _more_tags(self):
        """Legacy tags for Scikit-learn < 1.6."""
        return {
            # If bounds are (0, 1) or positive, we can keep positive requirement.
            # If bounds include negatives (e.g. -1, 1), we should relax this.
            "requires_positive_X": self.bounds[0] >= 0,
            "requires_y": False,
            "X_types": ["1darray"],
        }

    def __sklearn_tags__(self):
        """New Tags API for Scikit-learn >= 1.6."""
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = self.bounds[0] >= 0
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = False
        return tags

    def fit(self, X, y=None, compute_normalization: bool = False):
        """
        Fit the Beta Kernel KDE model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            The training input samples. Values must be within the 'bounds' interval.
        y : Ignored
            Not used, present for API consistency by convention.
        compute_normalization : bool, default=False
            If True, the normalization constant C is computed via numerical 
            integration and stored in `self.normalization_constant_` immediately 
            after fitting. This makes subsequent calls to `pdf(normalized=True)` faster.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate Input
        X = check_array(X, ensure_2d=False, order='C')
        if X.ndim == 2:
            if X.shape[1] != 1:
                raise ValueError("Data must be 1D or a single column.")
            X = X.ravel()
            
        # Validate Domain Bounds
        lower, upper = self.bounds
        if lower >= upper:
            raise ValueError(f"Bounds must be strictly increasing. Got {self.bounds}")
            
        if not np.all((X >= lower) & (X <= upper)):
            raise ValueError(
                f"All data points must be within the interval {self.bounds}. "
                f"Found range [{X.min():.3f}, {X.max():.3f}]."
            )

        self.data_raw_ = X
        self.n_samples_ = len(X)
        
        # --- Transform Data to [0, 1] ---
        self.scale_factor_ = upper - lower
        self.shift_ = lower
        
        X_scaled = (X - self.shift_) / self.scale_factor_
        
        # Prepare clipped data (numerical stability for internal beta logic)
        self._epsilon = 1e-10
        self.data_clipped_ = np.clip(X_scaled, self._epsilon, 1.0 - self._epsilon)

        # Bandwidth Selection (Operates on scaled data)
        if isinstance(self.bandwidth, (float, int)) and not isinstance(self.bandwidth, bool):
             if self.bandwidth <= 0:
                 raise ValueError("Bandwidth must be positive.")
             self.bandwidth_ = float(self.bandwidth)
             self.is_fallback_ = False
        
        elif isinstance(self.bandwidth, str):
            if self.bandwidth not in self.VALID_SELECTION_METHODS:
                raise ValueError(f"Unknown bandwidth selection method: '{self.bandwidth}'")
            self._select_bandwidth(method=self.bandwidth)
            
        elif self.bandwidth is None:
            if self.verbose > 0:
                print("No bandwidth specified. Defaulting to MISE_rule.")
            self._select_bandwidth(method="MISE_rule")
            
        else:
            raise ValueError("Invalid bandwidth parameter.")

        self.is_fitted_ = True
        
        # Reset normalization constant on new fit
        self.normalization_constant_ = None
        
        # Optionally pre-compute normalization
        if compute_normalization:
            if self.verbose > 0:
                print("Computing normalization constant (numerical integration)...")
            self.get_normalization_constant()
            
        return self

    def _select_bandwidth(self, method):
        """Internal dispatcher for bandwidth selection."""
        if method == "LCV":
            if self.verbose > 0:
                print(f"Selecting bandwidth using LCV in {self.bandwidth_bounds}...")
            self.bandwidth_ = self._select_bandwidth_lcv(self.bandwidth_bounds)
            self.is_fallback_ = False
        
        elif method == "LSCV":
            if self.verbose > 0:
                print(f"Selecting bandwidth using LSCV in {self.bandwidth_bounds}...")
            self.bandwidth_ = self._select_bandwidth_lscv(
                self.bandwidth_bounds, 
                grid_points=self.selection_grid_points,
                heuristic_factor=self.heuristic_factor,
                integration_points=self.integration_points
            )
            self.is_fallback_ = False
            
        elif method == "MISE_rule":
            if self.verbose > 0:
                print("Selecting bandwidth using MISE rule...")
            self.bandwidth_, self.is_fallback_ = self._select_bandwidth_mise_rule()
            if self.verbose > 0:
                 if self.is_fallback_:
                     print(f"MISE rule failed constraints. Using fallback: h = {self.bandwidth_:.4f}")
                 else:
                     print(f"Bandwidth selected by MISE rule: h = {self.bandwidth_:.4f}")

    def get_normalization_constant(self) -> float:
        """
        Computes and caches the normalization constant C = integral(f_hat(x) dx).
        
        If the constant has already been computed (during fit or a previous call),
        returns the cached value. Otherwise, computes it via numerical integration
        and stores it for future use.
        
        Returns
        -------
        float
            The integral of the current density estimate over the bounds.
        """
        check_is_fitted(self)
        
        # Return cached value if available
        if getattr(self, "normalization_constant_", None) is not None:
            return self.normalization_constant_
        
        # Define a scalar function for quad integration (uses internal un-normalized score)
        def func(x_scaled):
            if x_scaled <= 0 or x_scaled >= 1:
                return 0.0
            x_c = np.clip(x_scaled, self._epsilon, 1.0 - self._epsilon)
            k_mat = self._kernel_matrix(np.array([x_c]), self.data_clipped_, self.bandwidth_)
            return np.mean(k_mat)

        # Integrate over [0, 1]
        integral, error = scipy.integrate.quad(func, 0, 1, epsabs=1e-4, limit=50)
        
        # Cache the result
        self.normalization_constant_ = integral
        return integral

    def score_samples(self, X, normalized: bool = False):
        """
        Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            New data to evaluate.
            
        normalized : bool, default=False
            If True, ensures the PDF integrates to exactly 1.0. 
            Uses cached normalization constant if available, otherwise computes it.

        Returns
        -------
        log_density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in X.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False, order='C')
        if X.ndim == 2:
            X = X.ravel()
            
        lower, upper = self.bounds
        
        # 1. Transform input to [0, 1] space
        X_scaled = (X - self.shift_) / self.scale_factor_

        # 2. Create safe copy for kernel calculation
        out_of_bounds = (X_scaled < 0) | (X_scaled > 1)
        X_safe = X_scaled.copy()
        X_safe[out_of_bounds] = 0.5 

        # 3. Calculate density on [0, 1] scale
        kernel_matrix = self._kernel_matrix(X_safe, self.data_clipped_, self.bandwidth_)
        sum_kernels_per_x = kernel_matrix.sum(axis=1)
        pdf_values_scaled = (1 / self.n_samples_) * sum_kernels_per_x
        
        # 4. Zero out boundaries
        pdf_values_scaled[out_of_bounds] = 0.0
        
        # 5. Apply Change of Variables Formula
        # Add tiny epsilon to avoid log(0)
        log_pdf_scaled = np.log(pdf_values_scaled + np.finfo(float).tiny)
        log_pdf = log_pdf_scaled - np.log(self.scale_factor_)
        
        if normalized:
            # This will use the cached value or compute+cache it now
            norm_const = self.get_normalization_constant()
            log_pdf -= np.log(norm_const)
            
        return log_pdf

    def pdf(self, X, normalized: bool = False):
        """
        Convenience method returning the probability density (exp(score_samples)).
        """
        is_scalar = np.isscalar(X) or (isinstance(X, np.ndarray) and X.ndim == 0)
        X_arg = np.array([X]) if is_scalar else X
        
        log_pdf = self.score_samples(X_arg, normalized=normalized)
        pdf_vals = np.exp(log_pdf)
        
        if is_scalar:
            return float(pdf_vals[0])
        return pdf_vals

    def plot(
        self,
        eval_points: np.ndarray = None,
        show_histogram: bool = True,
        bins: int = 20,
        normalized: bool = False,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the estimated Probability Density Function (PDF).
        """
        check_is_fitted(self)

        lower, upper = self.bounds
        
        eval_points = (
            np.linspace(lower, upper, 1000, endpoint=True)
            if eval_points is None
            else eval_points
        )

        if isinstance(eval_points, (int, float)):
            eval_points_arr = np.array([eval_points])
        else:
            eval_points_arr = eval_points

        pdf_values = self.pdf(eval_points_arr, normalized=normalized)

        created_ax = False
        fig: Optional[plt.Figure] = None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_ax = True
        else:
            fig = ax.figure

        plot_label = f"Beta KDE (h={self.bandwidth_:.4f})" if label is None else label
        if normalized:
            plot_label += " [Normalized]"
            
        ax.plot(
            eval_points,
            pdf_values,
            label=plot_label,
            **kwargs,
        )

        if show_histogram:
            ax.hist(
                self.data_raw_,
                bins=bins,
                density=True,
                alpha=0.5,
                label="Data Histogram",
                color="gray",
                edgecolor="black",
                range=(lower, upper),
            )

        ax.set_title("Beta Kernel Density Estimation")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(lower, upper)
        ax.set_ylim(bottom=0)

        if created_ax and fig is not None:
            return fig, ax
        else:
            return ax

    # ----------------------------------------------------------------------
    #  Mathematical Core (Unchanged, operates on [0,1])
    # ----------------------------------------------------------------------

    def _rho_vec(self, x_arr, bandwidth):
        h = bandwidth
        h_squared = h**2
        term2_sqrt_arg = 4 * h_squared**2 + 6 * h_squared + 2.25 - x_arr**2 - x_arr / h
        term2_sqrt_arg = np.maximum(term2_sqrt_arg, 0.0)
        return (2 * h_squared + 2.5) - np.sqrt(term2_sqrt_arg)

    def _kernel_matrix(self, x_eval, data_pts, bandwidth):
        n_eval = x_eval.shape[0]
        x_col = x_eval.reshape(n_eval, 1)
        h = bandwidth
        
        lower_thresh = 2 * h
        upper_thresh = 1 - (2 * h)

        alpha_params = x_col / h
        beta_params = (1 - x_col) / h

        lower_mask = x_col < lower_thresh
        alpha_params = np.where(lower_mask, self._rho_vec(x_col, h), alpha_params)

        upper_mask = x_col > upper_thresh
        beta_params = np.where(upper_mask, self._rho_vec(1 - x_col, h), beta_params)

        return beta_dist.pdf(data_pts[np.newaxis, :], alpha_params, beta_params)

    # ----------------------------------------------------------------------
    #  Selection Logic (Unchanged, operates on [0,1])
    # ----------------------------------------------------------------------

    def _lcv_objective(self, bandwidth):
        if not (0 < bandwidth < 1): return np.inf
        n = self.n_samples_
        data = self.data_clipped_
        
        K_mat = self._kernel_matrix(data, data, bandwidth)
        row_sums = K_mat.sum(axis=1)
        diag_elems = np.diag(K_mat)
        
        f_hat_loo = (row_sums - diag_elems) / (n - 1)
        f_hat_loo = np.maximum(f_hat_loo, 1e-10)
        return -np.sum(np.log(f_hat_loo))

    def _select_bandwidth_lcv(self, bounds):
        res = scipy.optimize.minimize_scalar(
            self._lcv_objective, bounds=bounds, method="bounded"
        )
        if res.success: return float(res.x)
        raise RuntimeError(f"LCV failed: {res.message}")

    def _lscv_objective(self, bandwidth, integration_points=200):
        if not (0 < bandwidth < 1): return np.inf
        data = self.data_clipped_
        n = self.n_samples_
        
        x_grid = np.linspace(1e-5, 1.0 - 1e-5, integration_points)
        K_grid = self._kernel_matrix(x_grid, data, bandwidth)
        pdf_grid = K_grid.mean(axis=1)
        term1 = scipy.integrate.trapezoid(pdf_grid**2, x_grid)
        
        K_data = self._kernel_matrix(data, data, bandwidth)
        term2 = (np.sum(K_data) - np.sum(np.diag(K_data))) * (-2 / (n * (n - 1)))
        return term1 + term2

    def _select_bandwidth_lscv(self, bounds, grid_points, heuristic_factor, integration_points):
        std_dev = np.std(self.data_clipped_, ddof=1)
        n = self.n_samples_
        
        search_bounds = bounds
        if std_dev > 1e-8:
            h_rule = 0.9 * std_dev * (n ** (-0.2))
            search_bounds = (
                max(bounds[0], h_rule / heuristic_factor),
                min(bounds[1], h_rule * heuristic_factor)
            )
            
        h_grid = np.linspace(search_bounds[0], search_bounds[1], grid_points)
        scores = [self._lscv_objective(h, integration_points=integration_points) for h in h_grid]
        best_h = h_grid[np.nanargmin(scores)]
        best_score = np.nanmin(scores)
        
        step = h_grid[1] - h_grid[0] if grid_points > 1 else 0.01
        refine_bounds = (max(bounds[0], best_h - step), min(bounds[1], best_h + step))
        
        res = scipy.optimize.minimize_scalar(
            lambda h: self._lscv_objective(h, integration_points=integration_points), 
            bounds=refine_bounds, 
            method="bounded"
        )
        
        if res.success and res.fun <= best_score:
            return float(res.x)
        return best_h

    # ----------------------------------------------------------------------
    #  MISE Rule Logic (Unchanged, operates on [0,1])
    # ----------------------------------------------------------------------

    def _estimate_beta_params(self, X_filtered):
        if X_filtered.size == 0:
            raise ValueError("No data strictly within (0, 1).")

        mean_x = np.mean(X_filtered)
        var_x = np.var(X_filtered)

        if var_x == 0:
            raise ValueError("Sample variance is zero.")
        if var_x >= mean_x * (1 - mean_x):
            raise ValueError("Sample variance is too large for Beta parameters.")

        common_factor = ((mean_x * (1 - mean_x)) / var_x) - 1
        alpha_hat = mean_x * common_factor
        beta_hat = (1 - mean_x) * common_factor

        if alpha_hat <= 0 or beta_hat <= 0:
            raise ValueError(f"Estimated parameters not positive: a={alpha_hat}, b={beta_hat}")

        self.ahat_ = alpha_hat
        self.bhat_ = beta_hat
        return alpha_hat, beta_hat

    @staticmethod
    def _skewness(a, b):
        numerator = 2 * (b - a) * np.sqrt(a + b + 1)
        denominator = (a + b + 2) * np.sqrt(a * b)
        return numerator / denominator

    @staticmethod
    def _kurtosis(a, b):
        numerator = 6 * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2))
        denominator = a * b * (a + b + 2) * (a + b + 3)
        return numerator / denominator

    @staticmethod
    def _variance(a, b):
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def _calculate_hybrid_fallback(self, a, b):
        s = np.sqrt(self._variance(a, b))
        sk = self._skewness(a, b)
        kurt = self._kurtosis(a, b)
        n = self.n_samples_

        correction_factor = 1 + abs(sk) + abs(kurt)
        C = s / correction_factor

        if s == 0: return 1e-5
        return C * (n ** (-0.4))

    def _select_bandwidth_mise_rule(self):
        # Operates on scaled data in [0, 1]
        X_filtered = self.data_clipped_[(self.data_clipped_ > 0) & (self.data_clipped_ < 1)]
        
        h_final = 0.1
        is_fallback = False

        try:
            ahat, bhat = self._estimate_beta_params(X_filtered)

            if not (ahat > 1.5 and bhat > 1.5 and (ahat + bhat) > 3):
                raise ValueError("Parameters too small for MISE rule.")

            a = ahat
            b = bhat
            n = self.n_samples_

            log_num = (
                np.log(2 * a + 2 * b - 5)
                + np.log(2 * a + 2 * b - 3)
                + sp.gammaln(2 * a + 2 * b - 6)
                + sp.gammaln(a)
                + sp.gammaln(b)
                + sp.gammaln(a - 0.5)
                + sp.gammaln(b - 0.5)
            )
            denom_term_1 = (a - 1) * (b - 1)
            denom_term_2 = 6 - 4 * b + a * (3 * b - 4)
            
            if denom_term_1 <= 0 or denom_term_2 <= 0:
                raise ValueError("Denominator factor non-positive.")
                
            log_denom = (
                np.log(denom_term_1)
                + np.log(denom_term_2)
                + sp.gammaln(2 * a - 3)
                + sp.gammaln(2 * b - 3)
                + sp.gammaln(a + b)
                + sp.gammaln(a + b - 1)
            )
            log_factor = np.log(2) + np.log(n) + 0.5 * np.log(np.pi)
            log_h = (2 / 5) * (log_num - log_denom - log_factor)
            h_final = np.exp(log_h)

            if not (0 < h_final < 1):
                raise ValueError("Calculated bandwidth outside (0, 1).")

            is_fallback = False

        except (ValueError, RuntimeError) as e:
            if not (hasattr(self, "ahat_") and hasattr(self, "bhat_")):
                try:
                    self._estimate_beta_params(X_filtered)
                except ValueError:
                     if np.var(X_filtered) == 0:
                         return 1e-5, True
                     return 0.1, True

            h_final = self._calculate_hybrid_fallback(self.ahat_, self.bhat_)
            is_fallback = True
            
            if self.verbose > 0:
                warnings.warn(f"MISE Rule failed: {e}. Using fallback.", RuntimeWarning)

        return h_final, is_fallback