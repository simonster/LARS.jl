# The covariance test for the lasso
#
# Copyright 2014 Simon Kornblith <simon@simonster.com> and others

export covtest

struct CovarianceTestPath{T}
    predictor::Vector{Int}
    drop_in_cov::Vector{T}
    p::Vector{T}
    errorvar::Float64
    estimated_errorvar::Bool
end

function Base.show(io::IO, path::CovarianceTestPath{T}) where {T}
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

function covtest(path::LARSPath, X::Matrix{T}, y::Vector{T}; errorvar::Float64=NaN) where {T<:BlasReal}
    path.method == :lasso || error("covtest requires a lasso path")
    length(path.steps) > 1 || error("only one lasso step in path")

    has_intercept = isdefined(path, :intercept)
    if has_intercept
        μX = mean(X, dims=1)
        X .= X .- μX
        μy = mean(y)
        y .= y .- μy
    end
    adj = 1 + path.lambda2
    coefs = path.coefs
    lambdas = path.lambdas
    steps = path.steps

    # For each knot, look at the next knot and compare the model fit
    # with and without the new predictor
    yhat = zeros(size(y, 1))
    beta = zeros(size(coefs, 1))
    intercept_dir = zero(T)

    predictor = [path.steps[1].added]
    sizehint!(predictor, length(steps))

    # For first knot
    mul!(yhat, X, view(coefs, :, 2))
    drop_in_cov = [dot(y, yhat)]
    sizehint!(drop_in_cov, length(steps)+1)

    nactive = 1

    # Subsequent knots
    for k = 2:min(length(steps), length(lambdas)-1)
        if steps[k].added == 0
            nactive -= length(steps[k].dropped)
            continue
        end
        push!(predictor, steps[k].added)

        # Fit with old active set
        last_ldiff = lambdas[k]-lambdas[k-1]
        ldiff = lambdas[k+1]-lambdas[k]

        s = ldiff / last_ldiff
        for i = 1:size(coefs, 1)
            beta[i] = coefs[i, k] + (coefs[i, k] - coefs[i, k-1]) * s
        end
        mul!(yhat, X, beta)

        ip1 = dot(y, yhat)

        # Fit with new active set
        mul!(yhat, X, view(coefs, :, min(k+1, size(coefs, 2))))
        ip2 = dot(y, yhat)

        push!(drop_in_cov, adj * (ip2 - ip1))
        nactive += 1
    end

    estimate_errorvar = isnan(errorvar)
    if estimate_errorvar
        size(X, 2) < size(X, 1) || error("p >= n; error variance must be specified")

        # gelsy! here instead of gels! because X may not be full rank
        beta, rank = LAPACK.gelsy!(copy(X), copy(y))
        mul!(yhat, X, beta)

        errorvar = 0.0
        for i = 1:length(yhat)
            errorvar += abs2(y[i] - yhat[i])
        end

        df = size(X, 1) - (rank + has_intercept)
        errorvar /= df

        drop_in_cov .*= 1/errorvar
        p = ccdf(FDist(2, df), drop_in_cov)
    else
        drop_in_cov .*= 1/errorvar
        p = ccdf(Exponential(1), drop_in_cov)
    end

    CovarianceTestPath(predictor, drop_in_cov, p, errorvar, estimate_errorvar)
end