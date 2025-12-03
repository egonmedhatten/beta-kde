# Mathematical Formulation

`beta-kde` implements the **Beta Kernel Density Estimator** to solve the boundary leakage problem inherent in standard Gaussian KDEs when data is strictly bounded in $[0, 1]$.

While Beta Kernel estimators were proposed by Chen (1999), they have historically lacked a fast, reliable bandwidth selector. This package implements the novel **Beta Reference Rule**, a closed-form solution that reduces bandwidth selection complexity from iterative optimization to $\mathcal{O}(1)$.

## 1. The Boundary Bias Problem

Standard Kernel Density Estimation (KDE) typically uses a symmetric Gaussian kernel. Because the Gaussian kernel assumes an unbounded support $(-\infty, \infty)$, it suffers from severe **boundary bias** near the endpoints.

In these regions, probability mass "leaks" outside the valid domain. Standard corrections like **Reflection** force the derivative to vanish at the boundaries ($\hat{f}'(0)=0$), introducing "shoulder artifacts" that misrepresent distributions with non-zero boundary slopes (e.g., exponential or power-law data).

## 2. The Beta Kernel Estimator ($\hat{f}_2$)

This package implements the **modified Beta Kernel Estimator** ($\hat{f}_2$) proposed by Chen (1999). Among the available asymmetric estimators, $\hat{f}_2$ is chosen because it minimizes bias more effectively than the standard Beta estimator ($\hat{f}_1$) and achieves a smaller optimal Mean Integrated Squared Error (MISE).

For a sample $X_1, \dots, X_n$, the estimator is defined as:

$$
\hat{f}_2(x) = \frac{1}{n} \sum_{i=1}^{n} K^*_{x, h}(X_i)
$$

Unlike standard kernels which have a fixed shape, the kernel $K^*_{x, h}$ is a **Boundary Beta Kernel**. Its definition changes depending on whether the evaluation point $x$ is in the interior of the domain or near the boundaries ($2h$):

$$
K^*_{x, h}(t) = 
\begin{cases} 
K_{\rho(x, h), \frac{1-x}{h}}(t) & \text{if } x \in [0, 2h) \quad \text{(Left Boundary)} \\
K_{\frac{x}{h}, \frac{1-x}{h}}(t) & \text{if } x \in [2h, 1-2h] \quad \text{(Interior)} \\
K_{\frac{x}{h}, \rho(1-x, h)}(t) & \text{if } x \in (1-2h, 1] \quad \text{(Right Boundary)}
\end{cases}
$$

Where $K_{\alpha, \beta}(t)$ is the standard Beta density function, and the boundary correction parameter $\rho(x, h)$ is defined as:

$$
\rho(x, h) = 2h^2 + 2.5 - \sqrt{4h^4 + 6h^2 + 2.25 - x^2 - \frac{x}{h}}
$$

### Advantages
1.  **Natural Support:** The kernel is strictly non-negative and naturally matches the data support.
2.  **Adaptivity:** The variance of the kernel decreases as $x$ approaches the boundaries, automatically reducing smoothing where data is naturally denser.
3.  **Optimal Convergence:** It is free from boundary bias and achieves the optimal $\mathcal{O}(h^2)$ bias convergence rate everywhere.

## 3. Bandwidth Selection: The Beta Reference Rule

The primary contribution of this package is the **Beta Reference Rule**, a fast, closed-form bandwidth selector.

Existing methods like Least Squares Cross-Validation (LSCV) are computationally expensive and notoriously unstable for Beta kernels, often leading to undersmoothed estimates.

Our approach minimizes the **Asymptotic Mean Integrated Squared Error (AMISE)** of a Beta reference distribution. By using Method-of-Moments estimates for the data parameters $\hat{a}$ and $\hat{b}$, we derive the optimal bandwidth $h_{ref}$:

$$
h_{ref} = \left( \frac{\sqrt{\pi}}{n} \cdot \frac{(2\hat{a}-3)(2\hat{b}-3) 2^{-2\hat{a}-2\hat{b}+5} (2\hat{a}+2\hat{b}-5)(2\hat{a}+2\hat{b}-3) \Gamma(2(\hat{a}+\hat{b}-3))}{(\hat{a}(3\hat{b}-4)-4\hat{b}+6)\Gamma(\hat{a}+\hat{b}-1)\Gamma(\hat{a}+\hat{b})} \right)^{2/5}
$$

This rule matches the accuracy of numerical optimization while being several orders of magnitude faster (approx. $10^{-4}$ seconds per fit).

## 4. The Fallback Heuristic

The asymptotic approximation used in the Beta Reference Rule is valid only when the reference parameters satisfy $\hat{a}, \hat{b} > 1.5$.

For **U-shaped** or **J-shaped** distributions (where mass concentrates at the boundaries), these integrals diverge. `beta-kde` automatically detects this condition and applies a robust **Fallback Heuristic**:

$$
h_{heur} = \frac{\sqrt{\text{Var}(\hat{a}, \hat{b})}}{1 + |\text{Skewness}(\hat{a}, \hat{b})| + |\text{Ex. Kurtosis}(\hat{a}, \hat{b})|} \cdot n^{-2/5}
$$

This heuristic scales the bandwidth based on data dispersion while penalizing for high skewness and kurtosis, preventing overfitting in "spiky" boundary regions.

## 5. Multivariate Estimation: Copulas

For multidimensional data $X \in [0, 1]^d$, simple product kernels often fail to capture complex dependencies. `beta-kde` utilizes a **Non-Parametric Copula** approach.

We decompose the joint density $f(x_1, \dots, x_d)$ using Sklar's Theorem:

$$
f(x_1, \dots, x_d) = c(F_1(x_1), \dots, F_d(x_d)) \times \prod_{j=1}^{d} f_j(x_j)
$$

Where:
* $f_j(x_j)$ are the marginal densities estimated using 1D Beta Kernels with the Beta Reference Rule.
* $c(\cdot)$ is the copula density capturing the dependence structure, modeled using a multivariate Product Beta Kernel estimator.

## 6. Mass Conservation

It is a known property of asymmetric kernel estimators that they do not strictly preserve unit probability mass in finite samples.

However, we demonstrate that the deviation converges to 0 at a rate of $\mathcal{O}(h)$. In practice, this deviation is negligible (typically < 1% for $n \ge 100$).

* By default, `score_samples(normalized=False)` returns the raw density for maximum speed.
* If strict probability definition is required, `score_samples(normalized=True)` computes the normalization constant via numerical integration.

## References

<!-- 1.  **Szabadváry, J. H., et al. (2025).** A Fast, Closed-Form Bandwidth Selector for the Beta Kernel Density Estimator. *[Journal Name]*. -->
1.  **Chen, S. X. (1999).** Beta kernel estimators for density functions. *Computational Statistics & Data Analysis*, 31(2), 131-145.