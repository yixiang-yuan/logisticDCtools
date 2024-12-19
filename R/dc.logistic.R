#' Title: Large scale high-dimensional logistic regression using the divide-and-conquer approach
#'
#' @description
#' This function runs large scale high-dimension logistic regression with the divide-and-conquer approach.
#' It still allows for penalty functions including LASSO, SCAD and MCP in estimation, and also allows for
#' debiasing operations for each local estimate. The active set is combined using majority voting, while
#' the final estimate is determined from the chosen aggregation method, which includes 'average', 'sparse',
#' and 'weighted'.
#'
#' @param X the design matrix
#' @param y the response vector
#' @param k the number of nodes to randomly and equally split the data
#' @param tau the thresholding parameter proportion to k deciding the number of votes needed to be in the final active set
#' @param penalty penalty method. Supported methods include 'lasso', 'SCAD' and 'MCP'
#' @param aggregate.method aggregation method. Supported methods include 'average', 'sparse' and 'weighted'
#' @param debias boolean parameter indicating whether to desparsify the estimate
#' @param ridge_lambda lambda parameter used in the regularized inverse of the debiasing step
#'
#' @returns a list with the active set, all the local and the combined final estimators, and the running time.
#' @import ncvreg parallel foreach doParallel
#' @export
#'
#' @examples
#' # example code
#' p <- 1000
#' n <- 1000
#' s <- 30
#' k <- 5
#' tau <- 0.2
#' #' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' beta <- sample(c(rep(10 / sqrt(s), s), rep(0, p - s)))
#' beta.active <- ifelse(beta != 0, 1, 0)
#' y <- ifelse(1 / (1 + exp(-X %*% beta)) > 0.5, 1, 0)
#' result <- dc.logistic(X, y, k, tau, penalty = "lasso")
#' print(result)

dc.logistic <- function(
    X,
    y,
    k,
    tau,
    penalty = "lasso",
    aggregate.method = "average",
    debias = F,
    ridge_lambda = 0.01
) {
  if (!(penalty %in% c("lasso", "MCP", "SCAD"))) {
    stop("Penalty should be one of \"lasso\", \"SCAD\" ,  \"MCP\" ")
  }
  if (!(aggregate.method %in% c("average", "sparse", "weighted"))) {
    stop("Aggregate method should be one of \"average\", \"sparse\", \"weighted\"")
  }

  n <- nrow(X)
  p <- ncol(X)

  # Randomly split the data
  nlist <- rep(1:k, ceiling(n / k) + 1)[1:n]
  shuffled <- sample(nlist, n)
  dataseq <- 1:n
  split.index <- lapply(1:k, function(x) {dataseq[shuffled==x]})

  # Start the clusters
  cl <- parallel::makeCluster(k)
  doParallel::registerDoParallel(cl)

  # Ensure that cluster is stopped on exit
  on.exit({
    parallel::stopCluster(cl)
    foreach::registerDoSEQ()
  }, add = TRUE)

  # Start the timer now (avoid overhead of starting and shutting down the cluster)
  start.time <- proc.time()

  # Get the active sets and local estimates
  res <- c(foreach(idx=split.index, .combine = list, .multicombine = T) %dopar% {
    X.local <- X[idx, ]
    y.local <- y[idx]

    # There may be some weird errors so set the estimate to 0 if it happens
    beta.local <- tryCatch(
      {
        model <- ncvreg::cv.ncvreg(X.local, y.local, family = "binomial", penalty = penalty)
        # Do not consider the intercept
        coef(model)[-1]
      },
      error = function(e) {
        return(rep(0, p))
      }
    )

    active <- ifelse(beta.local != 0, 1, 0)

    # Debias the estimator if needed
    if (debias) {
      # One-step Newton-style debiasing
      phat <- 1 / (1 + exp(- X.local %*% beta.local))
      score <- t(X.local) %*% (y.local - phat)
      w_hat <- diag(as.numeric(phat * (1-phat)))
      hessian <- t(X.local) %*% w_hat %*% X.local
      reg.hessian.inv <- solve(hessian + ridge_lambda * diag(p))
      beta.local <- beta.local + reg.hessian.inv %*% score
    }
    list(
      active = active,
      beta.local = beta.local
    )
  })

  active.vote <- apply(
    matrix(do.call(cbind, lapply(res, function(x) {x$active})), nrow = p),
    1,
    sum
  )
  betas.local <- matrix(
    do.call(cbind, lapply(res, function(x) {x$beta.local})),
    nrow = p
  )

  # active.vote <- apply(ifelse(betas.local != 0, 1, 0), 1, sum)
  activeset <- ifelse(active.vote >= ceiling(tau * k), 1, 0)

  if (aggregate.method == "average") {
    beta.final = c()
    for (i in 1:p) {
      beta.final <- c(beta.final, mean(betas.local[i, ]))
    }
  }
  else if (aggregate.method == "sparse") {
    beta.final <- colMeans(betas.local) * activeset
  }
  else {
    if (sum(activeset) > 0){
      denom <- foreach(i=1:k, .combine = "+") %dopar% {
        X.local <- X[split.index[[i]], ]
        X.masked <- X.local[, as.logical(activeset)]
        beta.local <- betas.local[, i]
        phat <- 1 / (1 + exp(- X.local %*% beta.local))
        w_hat <- diag(as.numeric(phat * (1-phat)))
        t(X.masked) %*% w_hat %*% X.masked
      }
      nom <- foreach(i=1:k, .combine = "+") %dopar% {
        X.local <- X[split.index[[i]], ]
        X.masked <- X.local[, as.logical(activeset)]
        beta.local <- betas.local[, i]
        phat <- 1 / (1 + exp(- X.local %*% beta.local))
        w_hat <- diag(as.numeric(phat * (1-phat)))
        t(X.masked) %*% w_hat %*% X.masked %*% beta.local[as.logical(activeset)]
      }
      beta.final <- diag(p)[, as.logical(activeset)] %*% solve(denom) %*% nom
    }
    else {
      beta.final <- rep(0, p)
    }

  }


  # Record time
  elapsed.time <- as.numeric((proc.time() - start.time)['elapsed'])

  return(list(
    final = beta.final,
    local = betas.local,
    active = activeset,
    time = elapsed.time
  ))
}
