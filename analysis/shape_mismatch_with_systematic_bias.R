library(Matrix)

x <- c(1.01, 0.99, 1.01, 0.99)
y <- c(0.99, 1.01, 0.99, 1.01)
z <- y / x

######################################################################
# Scenario 1: Introduce a systematic bias for each measurement type
######################################################################

S <- matrix(c(
  # row 1
  1, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0, 0,
  # row 2
  0, 0, 0, 0, 1, 0, 0, 0,
  0, 0, 0, 0, 0, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 0, 0, 1,
  # row 3
  1/x[1], 0, 0, 0, -x[1]/y[1]^2, 0, 0, 0,
  0, 1/x[2], 0, 0, 0, -x[1]/y[1]^2, 0, 0,
  0, 0, 1/x[3], 0, 0, 0, -x[1]/y[1]^2, 0,
  0, 0, 0, 1/x[4], 0, 0, 0, -x[4]/y[4]^2
), nrow=12, ncol=8, byrow=TRUE)

m <- c(
  1, 1, 1, 1,
  1, 1, 1, 1,
  1, 1, 1, 1
)

chancov <- matrix(0.01^2, nrow=4, ncol=4)
B <- bdiag(chancov, chancov, chancov)
B <- B + diag(rep(1e-10, nrow(B)))

sqrt(diag(solve(t(S) %*% solve(B) %*% S)))

######################################################################
# Scenario 2: Introduce a systematic (energy-dependent) bias using
#             Matern covariance matrix for each measurement type
######################################################################

matern_cov32 <- function(x, s, r) {
  d <- abs(outer(x, x, `-`))
  z1 <- 1 + sqrt(3) * d/r
  z2 <- exp(-sqrt(3)*d/r)
  covmat <- s^2 * z1 * z2
}

chancov <- matern_cov32(c(1,2,3,4), 0.01, 10)
B <- bdiag(chancov, chancov, chancov)
B <- B + diag(rep(1e-10, nrow(B)))

sqrt(diag(solve(t(S) %*% solve(B) %*% S)))
chancov
