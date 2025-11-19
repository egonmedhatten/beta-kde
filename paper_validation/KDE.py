import warnings  # Import warnings for the fallback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np  # <-- Make sure numpy is imported
import scipy.integrate
import scipy.optimize
import scipy.special as sp
from scipy.stats import beta as beta_dist

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class BetaKernelKDE:
    """
    A class for performing Kernel Density Estimation (KDE)
    using the Beta kernel as proposed by Chen (1999), including boundary correction.
    Supports various bandwidth selection methods.
    """

    VALID_SELECTION_METHODS: List[str] = ["LCV", "LSCV", "MISE_rule"]

    # --- Attributes set during/after fit ---
    data_: Optional[np.ndarray] = None
    data_clipped_: Optional[np.ndarray] = None  # <-- MODIFICATION: Added clipped data
    n_samples_: int = 0
    bandwidth: Optional[float] = None
    is_fallback: bool = False

    # --- Attributes set during init ---
    verbose: int
    _bandwidth_param: Optional[Union[float, str]]
    _epsilon: float = 1e-10

    def __init__(self, bandwidth: Optional[Union[float, str]] = None, verbose: int = 1):
        """
        Initializes the BetaKernelKDE estimator.
        """
        self._bandwidth_param = bandwidth
        self.verbose = verbose

        if isinstance(bandwidth, (int, float)):
            if bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            self.bandwidth = float(bandwidth)
        elif isinstance(bandwidth, str):
            if bandwidth not in self.VALID_SELECTION_METHODS:
                raise ValueError(
                    f"Unknown bandwidth selection method: '{bandwidth}'. "
                    f"Choose from {self.VALID_SELECTION_METHODS}."
                )
        elif bandwidth is not None:
            raise TypeError("Bandwidth must be a float, string, or None.")

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """
        Validates the input data array.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data must be a numpy array.")
        if X.ndim > 2 or (X.ndim == 2 and X.shape[1] != 1):
            raise ValueError(
                "Data must be a 1D array or a 2D array with a single column."
            )
        if X.ndim == 2:
            X = X.flatten()
        if not np.all((X >= 0) & (X <= 1)):
            raise ValueError(
                "All data points must be within the [0, 1] interval for the Beta kernel."
            )
        return X

    def fit(
        self,
        X: np.ndarray,
        bandwidth: Optional[Union[float, str]] = None,
        bandwidth_selection_method: Optional[str] = None,
        bandwidth_bounds: Tuple[float, float] = (0.01, 0.2),
        **selection_kwargs: Any,
    ) -> "BetaKernelKDE":
        """
        Fits the Beta KDE model to the provided data and, if no bandwidth
        is set, performs bandwidth selection.
        """
        X = self._validate_data(X)

        # --- MODIFICATION: Store both raw and clipped data ---
        # self.data_ is the ORIGINAL, un-clipped data
        self.data_ = X
        # self.data_clipped_ is for LCV/LSCV to prevent crashes on 0s/1s
        self.data_clipped_ = np.clip(X, self._epsilon, 1.0 - self._epsilon)
        # --- END OF MODIFICATION ---

        self.n_samples_ = len(self.data_)
        self.is_fallback = False  # Reset fallback state on every fit
        self.bandwidth = bandwidth if bandwidth is not None else self.bandwidth

        effective_selection_method = bandwidth_selection_method
        if effective_selection_method is None and isinstance(
            self._bandwidth_param, str
        ):
            effective_selection_method = self._bandwidth_param

        if self.bandwidth is not None:
            if effective_selection_method is not None and self.verbose > 0:
                print(
                    f"Warning: A numeric bandwidth (h={self.bandwidth}) was provided. Ignoring selection method '{effective_selection_method}'."
                )
            return self

        if self.bandwidth is None:
            if effective_selection_method == "LCV":
                if self.verbose > 0:
                    print(
                        f"No bandwidth provided. Starting LCV bandwidth selection within bounds {bandwidth_bounds}..."
                    )
                self.bandwidth = self.select_bandwidth_lcv(
                    bounds=bandwidth_bounds, **selection_kwargs
                )
                if self.verbose > 0:
                    print(
                        f"Optimal bandwidth selected by LCV: h = {self.bandwidth:.4f}"
                    )

            elif effective_selection_method == "LSCV":
                if self.verbose > 0:
                    print(
                        f"No bandwidth provided. Starting LSCV bandwidth selection within bounds {bandwidth_bounds}..."
                    )
                self.bandwidth = self.select_bandwidth_lscv(
                    bounds=bandwidth_bounds, **selection_kwargs
                )
                if self.verbose > 0:
                    print(
                        f"Optimal bandwidth selected by LSCV: h = {self.bandwidth:.4f}"
                    )

            elif effective_selection_method == "MISE_rule":
                if self.verbose > 0:
                    print("No bandwidth provided. Applying MISE rule of thumb...")
                self.bandwidth, self.is_fallback = self.select_bandwidth_mise_rule()
                if self.verbose > 0:
                    if self.is_fallback:
                        print(
                            f"MISE rule failed constraints. Using fallback bandwidth: h = {self.bandwidth:.4f}"
                        )
                    else:
                        print(
                            f"Bandwidth selected by MISE rule: h = {self.bandwidth:.4f}"
                        )

            elif effective_selection_method is not None:
                raise ValueError(
                    f"Unknown bandwidth selection method: '{effective_selection_method}'. Choose from {self.VALID_SELECTION_METHODS}."
                )

        if self.bandwidth is None:
            raise RuntimeError(
                "No bandwidth was provided during initialization, and no selection method was specified "
                "or selection failed. Please provide a bandwidth or specify a valid selection method."
            )

        return self

    def _rho(self, x: float, bandwidth: float) -> float:
        """
        Calculates the rho function for boundary correction as defined in Chen (1999).
        """
        h_squared = bandwidth**2
        term2_sqrt_arg = 4 * h_squared**2 + 6 * h_squared + 2.25 - x**2 - x / bandwidth
        if term2_sqrt_arg < 0:
            term2_sqrt_arg = 0.0
        return (2 * h_squared + 2.5) - np.sqrt(term2_sqrt_arg)

    def _rho_vec(self, x_arr: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Vectorized version of the _rho boundary correction function.
        """
        h = bandwidth
        h_squared = h**2
        term2_sqrt_arg = 4 * h_squared**2 + 6 * h_squared + 2.25 - x_arr**2 - x_arr / h
        term2_sqrt_arg = np.maximum(term2_sqrt_arg, 0.0)
        return (2 * h_squared + 2.5) - np.sqrt(term2_sqrt_arg)

    def _beta_pdf(self, val: float, alpha: float, beta: float) -> float:
        """
        Computes the PDF of a Beta distribution at a given value.
        """
        if not (0 < val < 1):
            return 0.0
        if alpha <= 0 or beta <= 0:
            return 0.0
        try:
            log_beta_func = (
                sp.gammaln(alpha) + sp.gammaln(beta) - sp.gammaln(alpha + beta)
            )
            log_pdf = (
                (alpha - 1) * np.log(val) + (beta - 1) * np.log(1 - val) - log_beta_func
            )
            return np.exp(log_pdf)
        except (FloatingPointError, ValueError):
            return 0.0

    def _kernel(self, x: float, data_point: float, bandwidth: float) -> float:
        """
        The boundary-corrected Beta kernel K*_{x,bw}(t) from Chen (1999).
        """
        if not (0 < x < 1):
            return 0.0
        lower_threshold = 2 * bandwidth
        upper_threshold = 1 - (2 * bandwidth)
        alpha_param: float
        beta_param: float
        if lower_threshold <= x <= upper_threshold:
            alpha_param = x / bandwidth
            beta_param = (1 - x) / bandwidth
        elif 0 <= x < lower_threshold:
            alpha_param = self._rho(x, bandwidth)
            beta_param = (1 - x) / bandwidth
        elif upper_threshold < x <= 1:
            alpha_param = x / bandwidth
            beta_param = self._rho(1 - x, bandwidth)
        else:
            return 0.0
        return self._beta_pdf(data_point, alpha_param, beta_param)

    def _kernel_matrix(
        self, x_eval: np.ndarray, data_pts: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """
        Calculates the (N_eval, N_data) matrix of boundary-corrected Beta kernels
        using vectorization.

        IMPORTANT: x_eval (evaluation points) and data_pts (data points)
        MUST be strictly within (0, 1) for this to be stable.
        """
        n_eval = x_eval.shape[0]
        x_col = x_eval.reshape(n_eval, 1)

        h = bandwidth
        lower_thresh = 2 * h
        upper_thresh = 1 - (2 * h)

        # 1. Calculate alpha, beta parameters
        alpha_params = x_col / h
        beta_params = (1 - x_col) / h

        lower_mask = x_col < lower_thresh
        alpha_params = np.where(lower_mask, self._rho_vec(x_col, h), alpha_params)

        upper_mask = x_col > upper_thresh
        beta_params = np.where(upper_mask, self._rho_vec(1 - x_col, h), beta_params)

        # 2. Evaluate the Beta PDF
        kernel_mat = beta_dist.pdf(data_pts[np.newaxis, :], alpha_params, beta_params)
        return kernel_mat

    def pdf(self, X_new: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
        """
        Estimates the Probability Density Function (PDF) for new data points
        using the boundary-corrected Beta kernel.
        """
        if self.data_ is None or self.n_samples_ == 0:
            raise RuntimeError(
                "The KDE model has not been fitted yet. Call .fit() first."
            )
        if self.bandwidth is None:
            raise RuntimeError("Bandwidth is not set.")

        is_scalar = False
        if isinstance(X_new, (int, float)):
            is_scalar = True
            X_new_arr = np.array([float(X_new)])
        elif isinstance(X_new, np.ndarray):
            X_new_arr = X_new
        else:
            raise TypeError("Input must be a numpy array, float, or int.")

        X_new_valid = self._validate_data(X_new_arr)

        # Clip evaluation points to be just inside (0, 1) to avoid NaNs
        X_new_clipped = np.clip(X_new_valid, self._epsilon, 1.0 - self._epsilon)

        # --- MODIFICATION: Use CLIPPED data for kernel evaluation ---
        # The data points (t_i) must also be clipped to avoid log(0) in the PDF.
        kernel_matrix = self._kernel_matrix(
            X_new_clipped, self.data_clipped_, self.bandwidth
        )
        # --- END MODIFICATION ---

        sum_kernels_per_x = kernel_matrix.sum(axis=1)
        pdf_values = (1 / self.n_samples_) * sum_kernels_per_x

        # Manually set 0s for points outside the [0, 1] range
        pdf_values = np.where((X_new_valid <= 0) | (X_new_valid >= 1), 0.0, pdf_values)

        return pdf_values[0] if is_scalar else pdf_values

    def _pdf_at_x_internal(self, x_val: float, bandwidth: float) -> float:
        """
        Calculates the PDF estimate at a single point x_val.
        """
        if not (0 < x_val < 1):
            return 0.0
        if self.data_ is None or self.n_samples_ == 0:
            return 0.0

        # --- MODIFICATION: Use CLIPPED data for kernel evaluation ---
        sum_kernels = 0.0
        for data_point in self.data_clipped_:
            sum_kernels += self._kernel(x_val, data_point, bandwidth)
        return (1 / self.n_samples_) * sum_kernels
        # --- END MODIFICATION ---

    def _lcv_objective(self, bandwidth: float) -> float:
        """
        Calculates the negative Leave-One-Out (LOO) log-likelihood (LCV objective).
        """
        if not (0 < bandwidth < 1):
            return np.inf
        if self.data_clipped_ is None:
            raise RuntimeError("LCV objective called before fitting data.")
        n_samples = self.n_samples_
        if n_samples < 2:
            return np.inf

        f_hat_at_data = np.zeros(n_samples)
        range_i = range(n_samples)
        if self.verbose >= 2 and tqdm is not None:
            range_i = tqdm(
                range_i,
                desc=f"LCV (h={bandwidth:.4f}) f_hat",
                total=n_samples,
                leave=False,
                unit="pt",
            )

        # --- MODIFICATION: Use CLIPPED data for stability ---
        data = self.data_clipped_
        for i in range_i:
            sum_kernels = 0.0
            for j in range(n_samples):
                sum_kernels += self._kernel(data[i], data[j], bandwidth)
            f_hat_at_data[i] = (1 / n_samples) * sum_kernels

        k_self = np.zeros(n_samples)
        for i in range(n_samples):
            k_self[i] = self._kernel(data[i], data[i], bandwidth)
        # --- END MODIFICATION ---

        log_likelihood_sum = 0.0
        for i in range(n_samples):
            numerator = n_samples * f_hat_at_data[i] - k_self[i]
            if numerator <= 1e-10:
                return np.inf
            log_likelihood_sum += np.log(numerator)
        return -log_likelihood_sum

    def _lscv_objective(self, bandwidth: float, integration_points: int) -> float:
        """
        Calculates the (vectorized) LSCV objective function.
        Uses a fixed-grid trapezoidal integration for Term 1 for speed.
        """
        if not (0 < bandwidth < 1):
            return np.inf
        if self.data_clipped_ is None:
            raise RuntimeError("LSCV objective called before fitting data.")
        n_samples = self.n_samples_
        if n_samples < 2:
            return np.inf

        # --- MODIFICATION: Use CLIPPED data for stability ---
        data = self.data_clipped_
        # --- END MODIFICATION ---

        # --- Term 1: Integral of f(x)^2 ---
        x_grid = np.linspace(1e-5, 1.0 - 1e-5, integration_points)

        kernel_mat_on_grid = self._kernel_matrix(x_grid, data, bandwidth)

        pdf_on_grid = (1 / n_samples) * kernel_mat_on_grid.sum(axis=1)
        integrand_values = pdf_on_grid**2
        term1 = scipy.integrate.trapezoid(integrand_values, x_grid)

        # --- Term 2: Sum of K_ij ---
        K_mat = self._kernel_matrix(data, data, bandwidth)

        kernel_sum = np.sum(K_mat) - np.sum(np.diag(K_mat))
        term2 = (-2 / (n_samples * (n_samples - 1))) * kernel_sum

        return term1 + term2

    def select_bandwidth_lcv(
        self, bounds: Tuple[float, float] = (0.01, 0.5), **kwargs: Any
    ) -> float:
        """
        Selects the optimal bandwidth using Likelihood Cross-Validation (LCV).
        """
        if self.data_ is None or self.n_samples_ == 0:
            raise RuntimeError("The KDE model has not been fitted yet.")
        objective_func: Callable[[float], float] = lambda h: self._lcv_objective(h)
        result = scipy.optimize.minimize_scalar(
            objective_func, bounds=bounds, method="bounded", **kwargs
        )
        if result.success:
            optimal_bandwidth = float(result.x)
            self.bandwidth = optimal_bandwidth
            return optimal_bandwidth
        else:
            raise RuntimeError(f"Bandwidth selection failed: {result.message}.")

    def select_bandwidth_lscv(
        self,
        bounds: Tuple[float, float] = (0.01, 0.5),
        grid_points: int = 30,
        heuristic_factor: float = 4.0,
        integration_points: int = 200,
        **kwargs: Any,
    ) -> float:
        """
        Selects the optimal bandwidth using a robust 3-stage LSCV.
        """
        if self.data_ is None or self.n_samples_ == 0:
            raise RuntimeError("The KDE model has not been fitted yet.")
        if grid_points < 2:
            raise ValueError("grid_points must be 2 or more.")
        if heuristic_factor <= 0:
            raise ValueError("heuristic_factor must be positive.")

        objective_func: Callable[[float], float] = lambda h: self._lscv_objective(
            h, integration_points=integration_points
        )

        # --- Stage 1: Define Search Bounds (n-dependent) ---
        n = self.n_samples_
        # --- MODIFICATION: Use CLIPPED data for std calculation for stability ---
        data_std = np.std(self.data_clipped_, ddof=1)
        # --- END MODIFICATION ---
        final_search_bounds = bounds

        if data_std > 1e-8:
            h_rule = 0.9 * data_std * (n ** (-0.2))
            dynamic_min = h_rule / heuristic_factor
            dynamic_max = h_rule * heuristic_factor
            search_min = max(bounds[0], dynamic_min)
            search_max = min(bounds[1], dynamic_max)

            if search_min < search_max:
                final_search_bounds = (search_min, search_max)

            if self.verbose > 0:
                print(f"LSCV: Heuristic h_rule={h_rule:.5f}")
                print(f"LSCV: Final search bounds={final_search_bounds}")
        elif self.verbose > 0:
            print(f"LSCV: Data std_dev is near zero. Using hard bounds {bounds}.")

        # --- Stage 3: Grid Search ---
        h_grid = np.linspace(
            final_search_bounds[0], final_search_bounds[1], grid_points
        )
        grid_step = h_grid[1] - h_grid[0] if grid_points > 1 else 0.0

        if self.verbose > 0:
            print(
                f"LSCV: Starting grid search with {grid_points} points in {final_search_bounds}..."
            )

        grid_scores = np.array([objective_func(h) for h in h_grid])

        if not np.any(np.isfinite(grid_scores)):
            raise RuntimeError(
                f"LSCV grid search failed: Objective function returned non-finite values {grid_scores}"
                " for all grid points. Try different bounds."
            )

        best_grid_idx = np.nanargmin(grid_scores)
        best_grid_h = h_grid[best_grid_idx]
        best_grid_score = grid_scores[best_grid_idx]

        if self.verbose > 0:
            print(
                f"LSCV: Grid search minimum found: h={best_grid_h:.5f} (score={best_grid_score:.5f})"
            )

        # --- Stage 4: Local Refinement ---
        refinement_lower = max(final_search_bounds[0], best_grid_h - grid_step)
        refinement_upper = min(final_search_bounds[1], best_grid_h + grid_step)
        refinement_bounds = (refinement_lower, refinement_upper)

        if self.verbose > 0:
            print(f"LSCV: Starting local refinement in bounds {refinement_bounds}...")

        result = scipy.optimize.minimize_scalar(
            objective_func,
            bounds=refinement_bounds,
            method="bounded",
            **kwargs,
        )

        if result.success and result.fun <= best_grid_score:
            optimal_bandwidth = float(result.x)
            if self.verbose > 0:
                print(
                    f"LSCV: Local refinement successful. Final h={optimal_bandwidth:.5f}"
                )
        else:
            optimal_bandwidth = best_grid_h
            if self.verbose > 0:
                if not result.success:
                    print(
                        f"LSCV: Local refinement failed ({result.message}). Using grid minimum."
                    )
                else:
                    print(
                        f"LSCV: Local refinement found worse score ({result.fun:.5f}). Using grid minimum."
                    )

        self.bandwidth = optimal_bandwidth
        return optimal_bandwidth

    def _estimate_beta_params(self, X_filtered: np.ndarray) -> Tuple[float, float]:
        """
        Estimates the alpha and beta parameters of a Beta distribution
        using the Method of Moments from the given data.

        NOTE: This MUST be called on data *strictly* within (0, 1).
        """
        # --- MODIFICATION: Check size of FILTERED data ---
        if X_filtered.size == 0:
            raise ValueError(
                "No data strictly within (0, 1). Cannot estimate Beta parameters."
            )
        # --- END MODIFICATION ---

        mean_x = np.mean(X_filtered)
        var_x = np.var(X_filtered)

        if var_x == 0:
            raise ValueError(
                "Sample variance is zero. Cannot estimate Beta parameters."
            )
        if var_x >= mean_x * (1 - mean_x):
            raise ValueError(
                "Sample variance is too large for valid Beta parameter estimation."
            )

        common_factor = ((mean_x * (1 - mean_x)) / var_x) - 1
        alpha_hat = mean_x * common_factor
        beta_hat = (1 - mean_x) * common_factor

        if alpha_hat <= 0 or beta_hat <= 0:
            raise ValueError(
                f"Estimated Beta parameters (alpha_hat={alpha_hat:.2f}, beta_hat={beta_hat:.2f}) are not positive."
            )

        self.ahat = alpha_hat
        self.bhat = beta_hat
        return alpha_hat, beta_hat

    @staticmethod
    def skewness(a, b):
        numerator = 2 * (b - a) * np.sqrt(a + b + 1)
        denominator = (a + b + 2) * np.sqrt(a * b)
        return numerator / denominator

    @staticmethod
    def kurtosis(a, b):
        numerator = 6 * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2))
        denominator = a * b * (a + b + 2) * (a + b + 3)
        return numerator / denominator

    @staticmethod
    def variance(a, b):
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def _calculate_hybrid_fallback(self, a, b) -> float:
        """
        Calculates the fallback bandwidth based on skewness, kurtosis, and
        standard deviation (computed from the estimated beta parameters).
        """
        if self.data_ is None or self.n_samples_ == 0:
            raise RuntimeError("Data must be fitted to calculate fallback.")

        s = np.sqrt(self.variance(a, b))
        sk = self.skewness(a, b)
        kurt = self.kurtosis(a, b)
        n = self.n_samples_

        correction_factor = 1 + abs(sk) + abs(kurt)
        C = s / correction_factor

        if s == 0:
            return 1e-5

        return C * (n ** (-0.4))

    def select_bandwidth_mise_rule(self) -> Tuple[float, bool]:
        """
        Selects the bandwidth using MISE rule.
        """
        if self.data_ is None or self.n_samples_ < 2:
            raise RuntimeError("Not enough data to run MISE rule (n < 2).")

        h_final: float
        is_fallback: bool

        # --- MODIFICATION: Define filtered data ONCE ---
        X_filtered = self.data_[(self.data_ > 0) & (self.data_ < 1)]
        # --- END MODIFICATION ---

        try:
            # --- MODIFICATION: Use FILTERED raw data ---
            ahat, bhat = self._estimate_beta_params(X_filtered)
            # --- END MODIFICATION ---

            if not (ahat > 1.5 and bhat > 1.5 and (ahat + bhat) > 3):
                raise ValueError(
                    f"Estimated Beta parameters (ahat={ahat:.4f}, bhat={bhat:.4f}) are too small "
                    "for the MISE rule. Requires ahat > 1.5, bhat > 1.5."
                )

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
                raise ValueError(
                    f"Denominator factor is non-positive (D1={denom_term_1:.4f}, D2={denom_term_2:.4f})."
                )
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
                raise ValueError(
                    f"Calculated bandwidth {h_final:.4f} is outside (0, 1)."
                )

            is_fallback = False

        except (ValueError, RuntimeError) as e:
            # --- MODIFICATION: Fix fallback logic ---
            # We must have ahat, bhat to calculate fallback
            if not (hasattr(self, "ahat") and hasattr(self, "bhat")):
                # Try to estimate again, this may fail
                try:
                    ahat, bhat = self._estimate_beta_params(X_filtered)
                except ValueError as e_inner:
                    raise RuntimeError(
                        f"MISE rule failed and could not estimate parameters for fallback: {e_inner}"
                    )

            h_final = self._calculate_hybrid_fallback(self.ahat, self.bhat)
            is_fallback = True

            if self.verbose > 0:
                warnings.warn(
                    f"MISE Rule failed: {e}. Using fallback rule. Optimality lost.",
                    RuntimeWarning,
                )
            # --- END MODIFICATION ---

        self.fallback = is_fallback
        self.bandwidth = h_final
        return h_final, is_fallback

    def plot(
        self,
        eval_points: np.ndarray = None,
        show_histogram: bool = True,
        bins: int = 20,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the estimated Probability Density Function (PDF).
        """
        if self.data_ is None or self.n_samples_ == 0:
            raise RuntimeError(
                "The KDE model has not been fitted yet. Call .fit() first."
            )
        if self.bandwidth is None:
            raise RuntimeError("Bandwidth is not set.")

        eval_points = (
            np.linspace(0, 1, 1000, endpoint=True)
            if eval_points is None
            else eval_points
        )

        if isinstance(eval_points, (int, float)):
            eval_points_arr = np.array([eval_points])
        else:
            eval_points_arr = eval_points

        pdf_values = self.pdf(eval_points_arr)

        created_ax = False
        fig: Optional[plt.Figure] = None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_ax = True
        else:
            fig = ax.figure

        plot_label = f"Beta KDE (h={self.bandwidth:.4f})" if label is None else label
        ax.plot(
            eval_points,
            pdf_values,
            label=plot_label,
            **kwargs,
        )

        if show_histogram:
            # --- MODIFICATION: Plot RAW data and set range ---
            ax.hist(
                self.data_,  # Use the original, unclipped data
                bins=bins,
                density=True,
                alpha=0.5,
                label="Data Histogram",
                color="gray",
                edgecolor="black",
                range=(0, 1),  # Ensure bins align to [0, 1]
            )
            # --- END MODIFICATION ---

        ax.set_title("Beta Kernel Density Estimation")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

        if created_ax and fig is not None:
            return fig, ax
        else:
            return ax


if __name__ == "__main__":
    import doctest

    print("Running doctests...")
    np.set_printoptions(legacy="1.13")
    results = doctest.testmod(verbose=False)

    if results.failed == 0:
        print(f"All {results.attempted} doctests passed.")
        print("-" * 30)
    else:
        print(f"!!! DOCTESTS FAILED: {results.failed} failures. Halting. !!!")
        exit()

    print("Running BetaKernelKDE example with plotting for all methods...")

    np.random.seed(42)
    data: np.ndarray = np.random.beta(a=3, b=5, size=100)

    # Add 0s and 1s to test clipping
    data[0] = 0.0
    data[1] = 1.0

    print(f"Generated {len(data)} data points from Beta(3, 5), with 0.0 and 1.0 added.")

    selection_methods: List[str] = ["MISE_rule", "LCV", "LSCV"]
    plot_styles: Dict[str, Dict[str, Any]] = {
        "MISE_rule": {"color": "blue", "linestyle": "-", "linewidth": 2},
        "LCV": {"color": "red", "linestyle": ":", "linewidth": 2.5},
        "LSCV": {"color": "purple", "linestyle": "-.", "linewidth": 2},
    }

    eval_points: np.ndarray = np.linspace(0.001, 0.999, 200)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        data,
        bins=25,
        density=True,
        alpha=0.4,
        label="Data Histogram",
        color="gray",
        edgecolor="black",
        range=(0, 1),  # <-- MODIFICATION: Add range to match
    )

    print("\nStarting bandwidth selection for all methods...")

    for method in selection_methods:
        print(f"\n--- Running: {method} ---")
        v_level: int = 2 if method in ["LCV", "LSCV"] else 1
        kde = BetaKernelKDE(bandwidth=method, verbose=v_level)
        bandwidth_bounds: Tuple[float, float] = (0.01, 0.4)

        try:
            kde.fit(
                data,
                bandwidth_bounds=bandwidth_bounds,
            )
            label: str = f"{method} (h={kde.bandwidth:.4f})"
            if method == "MISE_rule" and kde.is_fallback:
                label += " [Fallback]"

            style: Dict[str, Any] = plot_styles.get(method, {})
            kde.plot(eval_points, show_histogram=False, ax=ax, label=label, **style)

            if v_level == 1:
                print(f"Successfully plotted {method}.")

        except Exception as e:
            print(
                f"!!! SKIPPING {method}: Could not complete fitting or plotting. Error: {e}"
            )

    print("\nGenerating final combined plot...")
    ax.set_title("Beta Kernel Density Estimation (All Methods)")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    plt.show()
