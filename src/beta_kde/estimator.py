import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special as sp
from scipy.stats import beta as beta_dist
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_array, check_is_fitted

# TODO: Save the bandwidth_ attribute properly for multi-dimensional data. It should be a vector (or dictionary) 
# with the bandwidth for each marginal density. (currently, it is saved in self.marginal_bandwidths_)
# Ensure that we also save the marginal ahat_ and bhat_ (useful for debugging)


class BetaKDE(DensityMixin, BaseEstimator):
    r"""
    Beta Kernel Density Estimation with Scikit-learn API compatibility.

    This estimator is designed for data strictly bounded within a fixed support
    (default [0, 1]). It addresses the **Boundary Bias** problem common in
    Gaussian KDEs by using Beta distributions as kernels. These kernels naturally
    match the support of the data and adapt their shape (becoming asymmetric)
    near boundaries to prevent probability mass leakage [1]_.

    **Key Features:**

    - **1D Data:** Uses Chen's Boundary-Corrected Beta Kernel estimator.
    - **Bandwidth Selection:** Implements the "Beta Reference Rule" [2]_, a fast
      $\mathcal{O}(1)$ closed-form selector that matches the accuracy of iterative
      optimization (LSCV) while being orders of magnitude faster.
    - **Multivariate:** Uses a Non-Parametric Beta Copula. Marginals are fitted
      independently using the Beta Reference Rule, and dependence structure is
      modeled via a Product Beta Kernel on the transformed uniform space
      (Probability Integral Transform).

    Parameters
    ----------
    bandwidth : float, str, or None, default=None
        The bandwidth selection method for the MARGINALS.

        * **float**: A fixed bandwidth value ($0 < h < 1$) applied to all dimensions.
        * **'beta-reference'** (Default): The "Beta Reference Rule" proposed in [2]_.
          It minimizes the AMISE of a Beta reference distribution using method-of-moments.
          Includes a robust heuristic fallback for U-shaped or J-shaped distributions.
        * **'LCV'** (Likelihood Cross-Validation): Maximizes the likelihood of leave-one-out
          data. Very accurate but slow and potentially unstable.
        * **'LSCV'** (Least Squares Cross-Validation): Minimizes the integrated squared error.
          Good compromise between speed and accuracy, but computationally expensive.

    bounds : tuple of float, default=(0.0, 1.0)
        The strict support of the data (min, max). Data MUST be within this range.
        The estimator automatically scales data to [0, 1] internally and rescales
        results back to the original domain.

    bandwidth_bounds : tuple of float, default=(0.01, 0.2)
        The search range (min_h, max_h) used when `bandwidth` is set to 'LCV' or 'LSCV'.

    selection_grid_points : int, default=30
        Number of grid points to use for the brute-force stage of LSCV grid search.

    heuristic_factor : float, default=4.0
        Expansion factor for determining the heuristic search range in LSCV.

    integration_points : int, default=200
        Number of points used for numerical integration in LSCV optimization.

    copula_grid_size : int, default=1000
        Resolution of the grid used to transform Marginals to Copula space
        (Probability Integral Transform) for multivariate data.

    verbose : int, default=0
        Verbosity level. If > 0, prints bandwidth selection progress.

    Attributes
    ----------
    n_samples_ : int
        The number of samples in the training data.

    n_features_ : int
        The number of features (dimensions).

    bandwidth_ : float or None
        The selected bandwidth (if 1D). None if multivariate (see `marginal_bandwidths_`).

    marginal_bandwidths_ : list of float
        The selected bandwidth $h$ for each dimension.

    copula_bandwidth_ : float
        The bandwidth used for the Copula kernel (calculated via Scott's Rule),
        only applicable if n_features > 1.

    normalization_constant_ : float or None
        The computed integral of the density over the support. Used to normalize
        `pdf()` calls. Calculated lazily or upon request.

    training_data_ : array-like of shape (n_samples, n_features)
        The training data, stored to allow exact kernel scoring.

    Notes
    -----
    **Mathematical Formulation:**
    The density estimate $\hat{f}(x)$ is given by:

    .. math::
        \hat{f}(x) = \frac{1}{n} \sum_{i=1}^{n} K(x, X_i, h)

    where $K$ is the Beta density function with shape parameters dynamically
    adjusted based on $x$ and $h$ to minimize bias at the edges (Chen type 2).

    **Computational Complexity:**
    Prediction (scoring) is $\mathcal{O}(N_{train} \cdot N_{test})$ because this is an exact
    kernel method that does not use tree-based approximations. Performance may
    degrade for datasets larger than $N=50,000$.

    **Input Shape:**
    This estimator strictly enforces 2D input `(n_samples, n_features)` to ensure
    unambiguous behavior. For 1D arrays, you must use `X.reshape(-1, 1)`.

    References
    ----------
    .. [1] Chen, S. X. (1999). Beta kernel estimators for density functions.
           *Computational Statistics & Data Analysis*, 31(2), 131-145.
    .. [2] Szabadváry, J. H. (2025). A Fast, Closed-Form Bandwidth Selector for
           the Beta Kernel Density Estimator. *[Preprint/In Press]*.

    Examples
    --------
    >>> import numpy as np
    >>> from beta_kde.estimator import BetaKDE
    >>> # Generate synthetic bounded data
    >>> X = np.random.beta(2, 5, size=(100, 1))
    >>> # Fit the model using the fast Reference Rule
    >>> kde = BetaKDE(bounds=(0, 1), bandwidth='beta-reference')
    >>> kde.fit(X)
    >>> # Score new sample
    >>> log_dens = kde.score_samples(np.array([[0.5]]))
    """

    VALID_SELECTION_METHODS = ["LCV", "LSCV", "beta-reference"]

    def __init__(
        self,
        bandwidth: Optional[Union[float, str]] = None,
        bounds: Tuple[float, float] = (0.0, 1.0),
        bandwidth_bounds: Tuple[float, float] = (0.01, 0.2),
        selection_grid_points: int = 30,
        heuristic_factor: float = 4.0,
        integration_points: int = 200,
        copula_grid_size: int = 1000,
        verbose: int = 0,
    ):
        self.bandwidth = bandwidth
        self.bounds = bounds
        self.bandwidth_bounds = bandwidth_bounds
        self.selection_grid_points = selection_grid_points
        self.heuristic_factor = heuristic_factor
        self.integration_points = integration_points
        self.copula_grid_size = copula_grid_size
        self.verbose = verbose

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = self.bounds[0] >= 0
        # Enforce 2D to match Scikit-learn's standard API expectation
        tags.input_tags.one_d_array = False
        tags.input_tags.two_d_array = True
        tags.target_tags.required = False
        return tags

    def fit(self, X, y=None, compute_normalization: bool = False):
        """
        Fit the Beta Kernel Density model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be strictly 2D.
        y : Ignored
            Not used, present for API consistency by convention.
        compute_normalization : bool, default=False
            If True, computes the normalization constant (integral of pdf)
            immediately after fitting. Useful if you intend to call `pdf(normalized=True)`
            frequently.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Reset attributes
        self.bandwidth_ = None
        self.is_fallback_ = None
        self.normalization_constant_ = None

        # 1. Validate Input
        # ensure_2d=True forces standard sklearn behavior (ValueError on 1D input).
        X = check_array(X, ensure_2d=True, order="C", dtype=np.float64)

        self.n_samples_, self.n_features_ = X.shape
        self.training_data_ = X

        # REQUIRED for check_estimator compatibility
        self.n_features_in_ = self.n_features_

        # Validate Bounds
        lower, upper = self.bounds
        if lower >= upper:
            raise ValueError(f"Bounds must be strictly increasing. Got {self.bounds}")

        if not np.all((X >= lower) & (X <= upper)):
            raise ValueError(
                f"All data points must be within the interval {self.bounds}. "
                f"Found range [{X.min():.3f}, {X.max():.3f}]."
            )

        # Validate Bandwidth Parameter
        if isinstance(self.bandwidth, (float, int)) and not isinstance(
            self.bandwidth, bool
        ):
            if self.bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
        elif isinstance(self.bandwidth, str):
            if self.bandwidth not in self.VALID_SELECTION_METHODS:
                raise ValueError(
                    f"Unknown bandwidth selection method: '{self.bandwidth}'"
                )

        # 2. Scale Data to [0, 1]
        self.scale_factor_ = upper - lower
        self.shift_ = lower

        X_scaled = (X - self.shift_) / self.scale_factor_
        self._epsilon = 1e-10
        self.data_clipped_ = np.clip(X_scaled, self._epsilon, 1.0 - self._epsilon)

        # 3. Fit Marginals
        self.marginal_bandwidths_ = []
        fallback_statuses = []
        self.cdf_grids_ = []
        self.x_grids_ = []

        for d in range(self.n_features_):
            data_d = self.data_clipped_[:, d]

            if self.verbose > 0 and self.n_features_ > 1:
                print(f"Fitting Dimension {d+1}/{self.n_features_}...")

            h, is_fb = self._select_bandwidth_for_dim(data_d)
            self.marginal_bandwidths_.append(h)
            fallback_statuses.append(is_fb)

            if self.verbose > 0:
                if is_fb:
                    print(
                        f"  Dim {d+1}: MISE rule failed constraints. Using fallback: h = {h:.4f}"
                    )
                elif self.n_features_ > 1:
                    print(f"  Dim {d+1}: Bandwidth selected: h = {h:.4f}")

            # Pre-compute CDF for Copula transform
            if self.n_features_ > 1:
                grid = np.linspace(0, 1, self.copula_grid_size)
                log_pdf = self._score_samples_1d(grid, data_d, h)
                pdf = np.exp(log_pdf)
                cdf = np.cumsum(pdf)
                cdf = cdf / cdf[-1]  # Normalize
                self.x_grids_.append(grid)
                self.cdf_grids_.append(cdf)

        # Legacy attribute for 1D convenience
        if self.n_features_ == 1:
            self.bandwidth_ = self.marginal_bandwidths_[0]
            self.is_fallback_ = fallback_statuses[0]
            if self.verbose > 0:
                if self.is_fallback_:
                    print(
                        f"MISE rule failed constraints. Using fallback: h = {self.bandwidth_:.4f}"
                    )
                else:
                    print(f"Bandwidth selected by MISE rule: h = {self.bandwidth_:.4f}")

        # 4. Copula Bandwidth (Scott's Rule for U-space)
        if self.n_features_ > 1:
            self.U_train_ = self._transform_to_uniform(self.data_clipped_)
            # Scott's Rule: n^(-1 / (d + 4))
            self.copula_bandwidth_ = self.n_samples_ ** (-1.0 / (self.n_features_ + 4))

            if self.verbose > 0:
                print(f"Copula Bandwidth (Scott's Rule): {self.copula_bandwidth_:.4f}")

        self.is_fitted_ = True
        self.normalization_constant_ = None

        if compute_normalization:
            self.get_normalization_constant()

        return self

    def _normalization_integrand(self, x_val, h, data_d):
        """
        Integrand helper method. This is a class method to allow pickling,
        avoiding closure capture issues present in nested functions.
        """
        # Handle scalar input from quad
        if np.ndim(x_val) == 0:
            x_val = np.array([x_val])

        mask = (x_val > 0) & (x_val < 1)
        if not np.any(mask):
            return 0.0 if np.ndim(x_val) == 0 else np.zeros_like(x_val)

        x_valid = x_val[mask]
        k_mat = self._kernel_matrix(x_valid, data_d, h)

        res = np.zeros_like(x_val)
        res[mask] = np.mean(k_mat, axis=1)

        return res if res.size > 1 else res.item()

    def get_normalization_constant(self) -> float:
        """
        Computes and caches the normalization constant C = integral(f_hat(x) dx).
        """
        check_is_fitted(self)

        if self.normalization_constant_ is not None:
            return self.normalization_constant_

        marginal_constants = []
        for d in range(self.n_features_):
            h = self.marginal_bandwidths_[d]
            data_d = self.data_clipped_[:, d]

            # Pass arguments explicitly to allow pickling
            integral, _ = scipy.integrate.quad(
                self._normalization_integrand,
                0,
                1,
                args=(h, data_d),
                epsabs=1e-4,
                limit=50,
            )
            marginal_constants.append(integral)

        # Total constant is the product (Copula integrates to ~1)
        total_constant = np.prod(marginal_constants)
        self.normalization_constant_ = total_constant
        return total_constant

    def score_samples(self, X, normalized: bool = False):
        """
        Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score. Must be strictly 2D.
        normalized : bool, default=False
            If True, subtracts log(normalization_constant) to ensure the
            density integrates to 1.0.

        Returns
        -------
        log_density : ndarray of shape (n_samples,)
            Log-likelihood of each data point.
        """
        check_is_fitted(self)

        # Enforce 2D to match fit behavior
        if np.ndim(X) == 0:
            X = np.array([[X]])
        X = check_array(X, ensure_2d=True, order="C", dtype=np.float64)

        if hasattr(self, "n_features_in_"):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but BetaKDE is expecting "
                    f"{self.n_features_in_} features as input."
                )
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"Mismatch in dimensions. Model: {self.n_features_}, Data: {X.shape[1]}"
            )

        X_scaled = (X - self.shift_) / self.scale_factor_
        X_safe = np.clip(X_scaled, self._epsilon, 1.0 - self._epsilon)

        n_test = X.shape[0]
        log_density = np.zeros(n_test)

        # 1. Add Marginal Log-Likelihoods
        for d in range(self.n_features_):
            h = self.marginal_bandwidths_[d]
            train_d = self.data_clipped_[:, d]

            log_pdf_scaled = self._score_samples_1d(X_safe[:, d], train_d, h)
            # Jacobian correction for scaling: log(pdf(x)) = log(pdf(z)) - log(scale)
            log_pdf = log_pdf_scaled - np.log(self.scale_factor_)
            log_density += log_pdf

        # 2. Add Copula Log-Likelihood (If Multivariate)
        if self.n_features_ > 1:
            U_test = self._transform_to_uniform(X_safe)
            log_copula = self._score_copula(
                U_test, self.U_train_, self.copula_bandwidth_
            )
            log_density += log_copula

        if normalized:
            norm_const = self.get_normalization_constant()
            log_density -= np.log(norm_const)

        return log_density

    def score(self, X, y=None):
        """
        Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.
        y : Ignored

        Returns
        -------
        score : float
            Total log-likelihood.
        """
        return np.sum(self.score_samples(X))

    def pdf(self, X, normalized: bool = False):
        """
        Convenience method returning the probability density (exp(score_samples)).

        Parameters
        ----------
        X : array-like or scalar
            Data to score. Scalars are converted to 2D arrays internally.
        normalized : bool, default=False
            Whether to normalize the density.

        Returns
        -------
        pdf : array-like or float
            Probability density values.
        """
        is_scalar = np.ndim(X) == 0
        if is_scalar:
            X_arg = np.array([[X]])
        elif np.ndim(X) == 1:
            # Convenience reshape for PDF method only
            X_arg = X.reshape(-1, 1)
        else:
            X_arg = X

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
        ax: Optional[Any] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Any, Tuple[Any, Any]]:
        """
        Plots the estimated Marginal Probability Density Functions (PDFs).

        Parameters
        ----------
        eval_points : array-like, optional
            Points at which to evaluate the PDF. If None, creates a grid of 1000 points.
        show_histogram : bool, default=True
            Whether to plot the histogram of training data underneath the curve.
        bins : int, default=20
            Number of bins for the histogram.
        normalized : bool, default=False
            Whether to normalize the PDF for plotting.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        label : str, optional
            Label for the plot legend.
        **kwargs
            Additional keyword arguments passed to `ax.plot`.

        Returns
        -------
        ax : matplotlib.axes.Axes or array of Axes
            The axes containing the plot.
        """
        # Lazy import to avoid hard dependency on top-level
        import matplotlib.pyplot as plt

        check_is_fitted(self)
        lower, upper = self.bounds
        n_dims = self.n_features_

        if ax is None:
            if n_dims == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes_list = [ax]
            else:
                cols = int(np.ceil(np.sqrt(n_dims)))
                rows = int(np.ceil(n_dims / cols))
                fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                axes_list = axs.flatten()
        else:
            if n_dims == 1:
                try:
                    fig = ax.figure
                except AttributeError:
                    fig = None
                axes_list = [ax]
            else:
                if not isinstance(ax, (list, np.ndarray)):
                    warnings.warn(
                        "Multivariate plot requested but single axis provided. Plotting 1st dimension only."
                    )
                    axes_list = [ax]
                else:
                    axes_list = np.array(ax).flatten()
                    try:
                        fig = axes_list[0].figure
                    except AttributeError:
                        fig = None

        for d in range(len(axes_list)):
            if d >= n_dims:
                axes_list[d].axis("off")
                continue

            curr_ax = axes_list[d]
            data_d = self.training_data_[:, d]
            h = self.marginal_bandwidths_[d]

            if eval_points is None:
                x_plot = np.linspace(lower, upper, 1000)
            else:
                x_plot = eval_points

            x_scaled = (x_plot - self.shift_) / self.scale_factor_
            x_safe = np.clip(x_scaled, self._epsilon, 1.0 - self._epsilon)
            train_d = self.data_clipped_[:, d]

            log_pdf_scaled = self._score_samples_1d(x_safe, train_d, h)
            log_pdf = log_pdf_scaled - np.log(self.scale_factor_)
            pdf_vals = np.exp(log_pdf)

            if normalized:
                integral = np.trapz(pdf_vals, x_plot)
                if integral > 0:
                    pdf_vals /= integral

            if n_dims == 1:
                plot_label = f"Beta KDE (h={h:.3f})" if label is None else label
                curr_ax.set_title("Beta Kernel Density Estimation")
            else:
                plot_label = f"Dim {d+1} (h={h:.3f})" if label is None else label
                curr_ax.set_title(f"Dimension {d+1}")

            if normalized:
                plot_label += " [Norm]"

            curr_ax.plot(x_plot, pdf_vals, label=plot_label, **kwargs)

            if show_histogram:
                curr_ax.hist(
                    data_d,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    color="gray",
                    edgecolor="none",
                    range=(lower, upper),
                )

            curr_ax.set_xlim(lower, upper)
            curr_ax.set_ylim(bottom=0)
            curr_ax.legend()

        if fig is not None:
            plt.tight_layout()
            return fig, axes_list if n_dims > 1 else axes_list[0]
        return axes_list if n_dims > 1 else axes_list[0]

    def _select_bandwidth_for_dim(self, data_1d):
        method = self.bandwidth if self.bandwidth else "beta-reference"
        if isinstance(method, (float, int)):
            return float(method), False
        if method == "LCV":
            return self._select_bandwidth_lcv(data_1d, self.bandwidth_bounds), False
        elif method == "LSCV":
            return (
                self._select_bandwidth_lscv(
                    data_1d,
                    self.bandwidth_bounds,
                    self.selection_grid_points,
                    self.heuristic_factor,
                    self.integration_points,
                ),
                False,
            )
        elif method == "beta-reference":
            return self._select_bandwidth_beta_reference(data_1d)
        raise ValueError(f"Unknown method: {method}")

    def _score_samples_1d(self, x_eval, data_train, h):
        k_mat = self._kernel_matrix(x_eval, data_train, h)
        pdf_vals = np.mean(k_mat, axis=1)
        return np.log(pdf_vals + 1e-300)

    def _transform_to_uniform(self, X_scaled):
        U = np.zeros_like(X_scaled)
        for d in range(self.n_features_):
            U[:, d] = np.interp(X_scaled[:, d], self.x_grids_[d], self.cdf_grids_[d])
        return np.clip(U, 1e-5, 1 - 1e-5)

    def _score_copula(self, U_test, U_train, h):
        n_test = U_test.shape[0]
        n_train = U_train.shape[0]
        d_dims = U_test.shape[1]
        log_weights = np.zeros((n_test, n_train))
        for j in range(d_dims):
            k_mat_j = self._kernel_matrix(U_test[:, j], U_train[:, j], h)
            log_weights += np.log(k_mat_j + 1e-300)
        max_log = np.max(log_weights, axis=1)
        sum_exp = np.sum(np.exp(log_weights - max_log[:, None]), axis=1)
        log_copula = max_log + np.log(sum_exp + 1e-300) - np.log(n_train)
        return log_copula

    def _lcv_objective(self, bandwidth, data):
        if not (0 < bandwidth < 1):
            return np.inf
        n = len(data)
        K_mat = self._kernel_matrix(data, data, bandwidth)
        row_sums = K_mat.sum(axis=1)
        diag_elems = np.diag(K_mat)
        f_hat_loo = (row_sums - diag_elems) / (n - 1)
        f_hat_loo = np.maximum(f_hat_loo, 1e-10)
        return -np.sum(np.log(f_hat_loo))

    def _select_bandwidth_lcv(self, data, bounds):
        res = scipy.optimize.minimize_scalar(
            lambda h: self._lcv_objective(h, data), bounds=bounds, method="bounded"
        )
        if res.success:
            return float(res.x)
        raise RuntimeError("LCV failed")

    def _lscv_objective(self, bandwidth, data, integration_points):
        if not (0 < bandwidth < 1):
            return np.inf
        n = len(data)
        x_grid = np.linspace(1e-5, 1.0 - 1e-5, integration_points)
        K_grid = self._kernel_matrix(x_grid, data, bandwidth)
        pdf_grid = K_grid.mean(axis=1)
        term1 = scipy.integrate.trapezoid(pdf_grid**2, x_grid)
        K_data = self._kernel_matrix(data, data, bandwidth)
        term2 = (np.sum(K_data) - np.sum(np.diag(K_data))) * (-2 / (n * (n - 1)))
        return term1 + term2

    def _select_bandwidth_lscv(
        self, data, bounds, grid_points, heuristic_factor, integration_points
    ):
        std_dev = np.std(data, ddof=0)
        n = len(data)
        search_bounds = bounds
        if std_dev > 1e-8:
            h_rule = 0.9 * std_dev * (n ** (-0.2))
            search_bounds = (
                max(bounds[0], h_rule / heuristic_factor),
                min(bounds[1], h_rule * heuristic_factor),
            )
        h_grid = np.linspace(search_bounds[0], search_bounds[1], grid_points)
        scores = [self._lscv_objective(h, data, integration_points) for h in h_grid]
        best_h = h_grid[np.nanargmin(scores)]

        step = h_grid[1] - h_grid[0] if grid_points > 1 else 0.01
        refine_bounds = (max(bounds[0], best_h - step), min(bounds[1], best_h + step))

        res = scipy.optimize.minimize_scalar(
            lambda h: self._lscv_objective(h, data, integration_points),
            bounds=refine_bounds,
            method="bounded",
        )
        return float(res.x) if res.success else best_h

    def _select_bandwidth_beta_reference(self, data):
        X_filtered = data[(data > 0) & (data < 1)]
        h_final = 0.1
        is_fallback = False

        try:
            ahat, bhat = self._estimate_beta_params(X_filtered)
            if not (ahat > 1.5 and bhat > 1.5 and (ahat + bhat) > 3):
                raise ValueError("Parameters too small for MISE rule.")

            a, b, n = ahat, bhat, len(data)
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

        except (ValueError, RuntimeError) as e:
            # Allow fallback if sample variance is zero ONLY if N=1 or we decide to support it.
            # Currently strict on N>1 constant data to match tests.
            if ("Sample variance is zero" in str(e) or "too large" in str(e)) and len(
                data
            ) > 1:
                raise e

            if not (hasattr(self, "ahat_") and hasattr(self, "bhat_")):
                try:
                    self._estimate_beta_params(X_filtered)
                except ValueError:
                    return 1.0 * (len(data) ** (-0.4)), True
            h_final = self._calculate_hybrid_fallback(self.ahat_, self.bhat_, len(data))
            is_fallback = True
            if self.verbose > 0:
                warnings.warn(f"MISE Rule failed: {e}. Using fallback.", RuntimeWarning)
        return h_final, is_fallback

    def _estimate_beta_params(self, X_filtered):
        if X_filtered.size == 0:
            raise ValueError("No data strictly within (0, 1).")
        mean_x = np.mean(X_filtered)
        var_x = np.var(X_filtered, ddof=0)

        if var_x == 0:
            raise ValueError("Sample variance is zero.")
        if var_x >= mean_x * (1 - mean_x):
            raise ValueError("Sample variance is too large for Beta parameters.")

        common = ((mean_x * (1 - mean_x)) / var_x) - 1
        a, b = mean_x * common, (1 - mean_x) * common
        if a <= 0 or b <= 0:
            raise ValueError(f"Estimated parameters not positive: a={a}, b={b}")
        self.ahat_, self.bhat_ = a, b
        return a, b

    def _calculate_hybrid_fallback(self, a, b, n):
        s = np.sqrt(self._variance(a, b))
        correction = 1 + abs(self._skewness(a, b)) + abs(self._kurtosis(a, b))
        return (s / correction) * (n ** (-0.4)) if s > 0 else 1e-5

    @staticmethod
    def _skewness(a, b):
        return (2 * (b - a) * np.sqrt(a + b + 1)) / ((a + b + 2) * np.sqrt(a * b))

    @staticmethod
    def _kurtosis(a, b):
        num = 6 * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2))
        den = a * b * (a + b + 2) * (a + b + 3)
        return num / den

    @staticmethod
    def _variance(a, b):
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def _rho_vec(self, x_arr, bandwidth):
        h = bandwidth
        term2 = np.maximum(4 * h**4 + 6 * h**2 + 2.25 - x_arr**2 - x_arr / h, 0.0)
        return (2 * h**2 + 2.5) - np.sqrt(term2)

    def _kernel_matrix(self, x_eval, data_pts, bandwidth):
        n_eval = x_eval.shape[0]
        x_col = x_eval.reshape(n_eval, 1)
        h = bandwidth
        lower_thresh, upper_thresh = 2 * h, 1 - 2 * h
        alpha = x_col / h
        beta_p = (1 - x_col) / h
        alpha = np.where(x_col < lower_thresh, self._rho_vec(x_col, h), alpha)
        beta_p = np.where(x_col > upper_thresh, self._rho_vec(1 - x_col, h), beta_p)
        return beta_dist.pdf(data_pts[np.newaxis, :], alpha, beta_p)
