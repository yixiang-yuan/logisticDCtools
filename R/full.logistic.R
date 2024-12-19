#' Title: High-dimensional logistic regression without splitting the data set
#'
#' @description
#' This function runs high-dimension logistic regression without splitting the data set. It allows for
#' penalty functions including LASSO, SCAD and MCP in estimation to tackle with the high-dimensionality,
#' and also allows for debiasing operations in the final estimate. The function will return the estimated
#' active set and the final estimate.
#'
#' @param x the design matrix
#' @param y the response vector
#' @param penalty penalty method. Supported methods include 'lasso', 'SCAD' and 'MCP'
#' @param debias boolean parameter indicating whether to desparsify the estimate
#' @param ridge_lambda lambda parameter used in the regularized inverse of the debiasing step
#'
#' @returns a list with the active set, the final estimate and running time of the function.
#' @import ncvreg
#' @export
#'
#' @examples
#' # example code
#' p <- 1000
#' n <- 1000
#' s <- 30
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' beta <- sample(c(rep(10 / sqrt(s), s), rep(0, p - s)))
#' beta.active <- ifelse(beta != 0, 1, 0)
#' y <- ifelse(1 / (1 + exp(-X %*% beta)) > 0.5, 1, 0)
#' result <- full.logistic(X, y, penalty = "lasso")
#' print(result)

full.logistic <- function(
    x,
    y,
    penalty = "lasso",
    debias = F,
    ridge_lambda = 0.01
) {
  if (!(penalty %in% c("lasso", "MCP", "SCAD"))) {
    stop("Penalty should be one of \"lasso\", \"SCAD\" ,  \"MCP\" ")
  }

  n <- nrow(X)
  p <- ncol(X)

  # Start the timer now
  start.time <- proc.time()

  # Logistic procedure
  model <- ncvreg::cv.ncvreg(X, y, family = "binomial", penalty = penalty)
  # Do not consider the intercept
  beta.final <- coef(model)[-1]
  activeset <- ifelse(beta.final != 0, 1, 0)

  # Debias the estimator if needed
  if (debias) {
    # One-step Newton-style debiasing
    phat <- 1 / (1 + exp(- X %*% beta.final))
    score <- t(X) %*% (y - phat)
    w_hat <- diag(as.numeric(phat * (1-phat)))
    hessian <- t(X) %*% w_hat %*% X
    reg.hessian.inv <- solve(hessian + ridge_lambda * diag(p))
    beta.final <- beta.final + reg.hessian.inv %*% score
  }

  # Record time
  elapsed.time <- as.numeric((proc.time() - start.time)['elapsed'])

  return(list(
    final = beta.final,
    active = activeset,
    time = elapsed.time
  ))

}
