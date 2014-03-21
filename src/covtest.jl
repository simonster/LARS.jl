# The covariance test for the lasso
# 
# Copyright 2014 Simon Kornblith <simon@simonster.com> and others
#
# This program and the accompanying materials are made available under the
# terms of the GNU Lesser General Public License (LGPL) version 3.0 which
# accompanies this distribution, and is available at
# http://www.gnu.org/licenses/lgpl-3.0.html
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# See also:
# Lockhart, R., Taylor, J., Tibshirani, R. J., & Tibshirani, R. (2013). A
# significance test for the lasso. arXiv:1301.7161 [math, Stat]. Retrieved
# from http://arxiv.org/abs/1301.7161

export covtest

immutable CovarianceTestPath{T}
    predictor::Vector{Int}
    drop_in_cov::Vector{T}
    p::Vector{T}
    errorvar::Float64
    estimated_errorvar::Bool
end

function Base.show{T}(io::IO, path::CovarianceTestPath{T})
    println(io, "CovarianceTestPath{$T}:")
    println(io, " Error variance: $(path.errorvar) ",
            (path.estimated_errorvar ? "(estimated)" : "(given)"))

    spacing = repeat(" ", 2)
    predictorlen = max(iceil(log10(maximum(path.predictor))), 9)
    covlen = maximum([length(repr(x)) for x in path.drop_in_cov])
    print(io, ' ', rpad("Predictor", predictorlen), spacing,
          rpad("Drop in Covariance", covlen), spacing, rpad("p-value", 8))

    for i = 1:length(path.predictor)
        p = path.p[i]
        print(io, '\n', ' ', lpad(repr(path.predictor[i]), predictorlen), spacing,
              rpad(repr(path.drop_in_cov[i]), covlen), spacing,
              rpad((p < 1e-4 ? (@sprintf "%.1e" p) : (@sprintf "%.5f" p)), 8))
    end
end

function interceptdot{T}(y::Vector{T}, yhat::Vector{T}, intercept::T)
    r = zero(T)
    for i = 1:length(y)
        r += y[i]*(yhat[i]+intercept)
    end
    r
end

function covtest{T<:BlasReal}(path::LARSPath, X::Matrix{T}, y::Vector{T}; errorvar::Float64=NaN)
    path.method == :lasso || error("covtest requires a lasso path")
    length(path.steps) > 1 || error("only one lasso step in path")

    has_intercept = isdefined(path, :intercept)
    coefs = path.coefs
    lambdas = path.lambdas
    steps = path.steps

    # For each knot, look at the next knot and compare the model fit
    # with and without the new predictor
    yhat = zeros(size(y, 1))
    beta = zeros(size(coefs, 1))
    intercept_dir = zero(T)

    predictor = [path.steps[1].added]
    sizehint(predictor, length(steps))

    # For first knot, we are comparing against an intercept-only fit
    ip1 = has_intercept ? sum(y)*path.intercept[1] : sum(y)
    
    A_mul_B!(yhat, X, view(coefs, :, 2))
    ip2 = has_intercept ? interceptdot(y, yhat, path.intercept[2]) : dot(y, yhat)

    drop_in_cov = [ip2 - ip1]
    sizehint(drop_in_cov, length(steps))

    # Subsequent knots
    k = 2
    for i = 2:length(steps)-1
        if steps[i].added == 0
            k += 2
            continue
        end
        push!(predictor, steps[i].added)

        last_ldiff = lambdas[k]-lambdas[k-1]
        ldiff = lambdas[k+1]-lambdas[k]

        # Fit with old active set
        s = ldiff / last_ldiff
        for i = 1:size(coefs, 1)
            beta[i] = coefs[i, k] + (coefs[i, k] - coefs[i, k-1]) * s
        end
        A_mul_B!(yhat, X, beta)
        ip1 = if has_intercept
            local intercept = path.intercept[k] + (path.intercept[k] - path.intercept[k-1])*s
            interceptdot(y, yhat, intercept)
        else
            dot(y, yhat)
        end

        # Fit with new active set
        A_mul_B!(yhat, X, view(coefs, :, k+1))
        ip2 = has_intercept ? interceptdot(y, yhat, path.intercept[k]) : dot(y, yhat)

        push!(drop_in_cov, ip2 - ip1)
        k += 1
    end

    estimate_errorvar = isnan(errorvar)
    if estimate_errorvar
        size(X, 2) < size(X, 1) || error("p >= n; error variance must be specified")

        # XXX what to do if X is not full rank?
        df = size(X, 1) - (size(coefs, 2) + has_intercept)
        if lambdas[end] < eps()
            # Last lambda is least squares fit
            A_mul_B!(yhat, X, view(coefs, :, size(coefs, 2)))
            errorvar = zero(T)
            if has_intercept
                intercept = path.intercept[end]
            else
                intercept = zero(T)
            end
            for i = 1:length(y)
                @inbounds errorvar += abs2(y[i] - yhat[i] - intercept)
            end
            errorvar /= df
        else
            rank = size(coefs, 2)
            if has_intercept
                μX = mean(X, 1)
                Xcopy = X .- μX
                μy = mean(y)
                ycopy = y .- μy
            else
                Xcopy = copy(X)
                ycopy = copy(y)
            end
            errorvar = LAPACK.gels!('N', Xcopy, ycopy)[3][1]/df
        end
        scale!(drop_in_cov, 1/errorvar)
        p = ccdf(FDist(2, df), drop_in_cov)
    else
        scale!(drop_in_cov, 1/errorvar)
        p = ccdf(Exponential(1), drop_in_cov)
    end

    CovarianceTestPath(predictor, drop_in_cov, p, errorvar, estimate_errorvar)
end