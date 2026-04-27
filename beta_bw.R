#' Beta Kernel Rule-of-Thumb Bandwidth Selector
#'
#' A fast, closed-form bandwidth selector for the Beta Kernel Density Estimator.
#' Implements the MISE-optimal Beta Reference Rule with an automatic fallback
#' heuristic for U-shaped and J-shaped distributions.
#'
#' Reference:
#'   Hallberg Szabadváry, J. (2026). A Fast, Closed-Form Bandwidth Selector
#'   for the Beta Kernel Density Estimator. Journal of Computational and
#'   Graphical Statistics.
#'
#' @param x Numeric vector with values in [0, 1].
#' @return Scalar optimal bandwidth h > 0.

bw.beta.rot <- function(x) {
  # --- Input validation ---
  if (!is.numeric(x)) stop("'x' must be a numeric vector.")
  if (length(x) < 2L) stop("'x' must have at least 2 observations.")
  if (any(x < 0 | x > 1, na.rm = TRUE)) stop("All values in 'x' must be in [0, 1].")
  x <- x[!is.na(x)]
  n <- length(x)

  # --- Filter to strictly (0, 1) for parameter estimation ---
  x_f <- x[x > 0 & x < 1]
  if (length(x_f) == 0L) stop("No data strictly within (0, 1).")

  # --- Method-of-moments Beta parameter estimation ---
  mu <- mean(x_f)
  # Population variance (ddof = 0) to match the Python implementation
  v <- sum((x_f - mu)^2) / length(x_f)

  if (v == 0) stop("Sample variance is zero.")
  if (v >= mu * (1 - mu)) stop("Sample variance is too large for Beta parameters.")

  common <- (mu * (1 - mu)) / v - 1
  a <- mu * common
  b <- (1 - mu) * common
  if (a <= 0 || b <= 0) stop(sprintf("Estimated parameters not positive: a=%.4f, b=%.4f", a, b))

  # --- Bandwidth selection ---
  h <- NULL
  use_fallback <- FALSE

  if (a > 1.5 && b > 1.5 && (a + b) > 3) {
    # MISE-optimal Beta Reference Rule (log-space for numerical stability)
    log_num <- (
      log(2 * a + 2 * b - 5)
      + log(2 * a + 2 * b - 3)
      + lgamma(2 * a + 2 * b - 6)
      + lgamma(a)
      + lgamma(b)
      + lgamma(a - 0.5)
      + lgamma(b - 0.5)
    )

    denom_term_1 <- (a - 1) * (b - 1)
    denom_term_2 <- 6 - 4 * b + a * (3 * b - 4)

    if (denom_term_1 <= 0 || denom_term_2 <= 0) {
      use_fallback <- TRUE
    } else {
      log_denom <- (
        log(denom_term_1)
        + log(denom_term_2)
        + lgamma(2 * a - 3)
        + lgamma(2 * b - 3)
        + lgamma(a + b)
        + lgamma(a + b - 1)
      )

      log_factor <- log(2) + log(n) + 0.5 * log(pi)
      log_h <- (2 / 5) * (log_num - log_denom - log_factor)
      h <- exp(log_h)

      if (!(h > 0 && h < 1)) {
        use_fallback <- TRUE
        h <- NULL
      }
    }
  } else {
    use_fallback <- TRUE
  }

  # --- Fallback heuristic ---
  if (use_fallback) {
    beta_var  <- (a * b) / ((a + b)^2 * (a + b + 1))
    beta_skew <- (2 * (b - a) * sqrt(a + b + 1)) / ((a + b + 2) * sqrt(a * b))
    beta_kurt <- 6 * ((a - b)^2 * (a + b + 1) - a * b * (a + b + 2)) /
                 (a * b * (a + b + 2) * (a + b + 3))

    s <- sqrt(beta_var)
    if (s > 0) {
      correction <- 1 + abs(beta_skew) + abs(beta_kurt)
      h <- (s / correction) * n^(-0.4)
    } else {
      h <- 1e-5
    }
    warning("MISE Rule not applicable (shape parameters too small); using fallback heuristic.")
  }

  h
}


#--- Usage example ---
# set.seed(42)
# my_data <- rbeta(500, 2, 5)
# h_opt   <- bw.beta.rot(my_data)
# cat("Optimal bandwidth:", h_opt, "\n")

# library(kdensity)
# fit <- kdensity(my_data, kernel = "beta", bw = h_opt)
# plot(fit, main = "Beta KDE with Rule-of-Thumb Bandwidth")
