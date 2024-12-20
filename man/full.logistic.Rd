% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/full.logistic.R
\name{full.logistic}
\alias{full.logistic}
\title{Title: High-dimensional logistic regression without splitting the data set}
\usage{
full.logistic(x, y, penalty = "lasso", debias = F, ridge_lambda = 0.01)
}
\arguments{
\item{x}{the design matrix}

\item{y}{the response vector}

\item{penalty}{penalty method. Supported methods include 'lasso', 'SCAD' and 'MCP'}

\item{debias}{boolean parameter indicating whether to desparsify the estimate}

\item{ridge_lambda}{lambda parameter used in the regularized inverse of the debiasing step}
}
\value{
a list with the active set, the final estimate and running time of the function.
}
\description{
This function runs high-dimension logistic regression without splitting the data set. It allows for
penalty functions including LASSO, SCAD and MCP in estimation to tackle with the high-dimensionality,
and also allows for debiasing operations in the final estimate. The function will return the estimated
active set and the final estimate.
}
\examples{
# example code
p <- 1000
n <- 1000
s <- 30
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
beta <- sample(c(rep(10 / sqrt(s), s), rep(0, p - s)))
beta.active <- ifelse(beta != 0, 1, 0)
y <- ifelse(1 / (1 + exp(-X \%*\% beta)) > 0.5, 1, 0)
result <- full.logistic(X, y, penalty = "lasso")
print(result)
}
