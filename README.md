# Least angle regression

[![Build Status](https://travis-ci.org/simonster/LARS.jl.png)](https://travis-ci.org/simonster/LARS.jl)

## Introduction

[Least angle regression](http://en.wikipedia.org/wiki/Least-angle_regression)
is a variable selection/shrinkage procedure for high-dimensional data. It is
also an algorithm for efficiently finding all knots in the solution path for
the aforementioned this regression procedure, as well as for lasso
(L1-regularized) linear regression. Fitting the entire solution path is useful
for selecting the optimal value of the shrinkage parameter λ for a given
dataset, and for the [lasso covariance test](http://arxiv.org/abs/1301.7161),
which provides the significance of each variable addition along the lasso path.

## Usage

LARS solution paths are provided by the `lars` function:

```julia
lars(X, y; method=:lasso, intercept=true, standardize=true, use_gram=true,
     maxiter=typemax(Int), lambda_min=0.0, verbose=false)
```

`X` is the design matrix and `y` is the dependent variable. The optional parameters are:

`method` - either `:lasso` or `:lars`.

`intercept` - whether to fit an intercept in the model. The intercept is
always unpenalized.

`standardize` - whether to standardize the predictor matrix. In contrast to
linear regression, this affects the algorithm's results. The returned
coefficients are always unstandardized.

`use_gram` - whether to use a precomputed Gram matrix in computation.

`maxiter` - maximum number of iterations of the algorithm. If this is
exceeded, an incomplete path is returned. `lambda_min` - value of λ at which
the algorithm should stop.

`verbose` - if true, prints information at each step.

The `covtest` function computes the lasso covariance test based on a LARS path:

`covtest(path, X, y; errorvar)`

`path` is the output of the LARS function above, and `X` and `y` are the
independent and dependent variables used in fitting the path. If specified,
`errorvar` is the variance of the error. If not specified, the error variance
is computed based on the least squares fit of the full model.

## Notes

The output of `covtest` has minor discrepancies with that of the [covTest
package](http://cran.r-project.org/web/packages/covTest/index.html). This is
because the covTest package does not take into account the intercept in the
least squares model fit when computing the error variance, which I believe is
incorrect. I have emailed the authors but have yet to receive a response.

## Credits

This package is written and maintained by Simon Kornblith <simon@simonster.com>.

The `lars` function is derived from code from scikit-learn written by:
- Alexandre Gramfort <alexandre.gramfort@inria.fr>
- Fabian Pedregosa <fabian.pedregosa@inria.fr>
- Olivier Grisel <olivier.grisel@ensta.org>
- Vincent Michel <vincent.michel@inria.fr>
- Peter Prettenhofer <peter.prettenhofer@gmail.com>
- Mathieu Blondel <mathieu@mblondel.org>
- Lars Buitinck <L.J.Buitinck@uva.nl>
