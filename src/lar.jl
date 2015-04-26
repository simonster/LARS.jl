# Least angle regression
# 
# Copyright 2014 Simon Kornblith <simon@simonster.com> and others
#
# The LARS implementation here is heavily derived from scikit-learn.
#
# Copyright (c) 2007–2014 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#
# Original Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#                   Fabian Pedregosa <fabian.pedregosa@inria.fr>
#                   Olivier Grisel <olivier.grisel@ensta.org>
#                   Vincent Michel <vincent.michel@inria.fr>
#                   Peter Prettenhofer <peter.prettenhofer@gmail.com>
#                   Mathieu Blondel <mathieu@mblondel.org>
#                   Lars Buitinck <L.J.Buitinck@uva.nl>
#
# See also:
# Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle
# regression. The Annals of Statistics, 32(2), 407–499.
# doi:10.1214/009053604000000067

import Base.LinAlg.Ac_ldiv_B!, Base.LinAlg.BlasReal
export lars

immutable LARSStep
    added::Int
    dropped::@compat Tuple{Vararg{Int}}

    LARSStep(x::Int) = new(x)
    LARSStep(x::@compat Tuple{Vararg{Int}}) = new(0, x)
end

immutable LARSPath{T}
    # :lasso or :lar
    method::Symbol

    # Whether the predictors were standardized
    standardized::Bool

    # The value of the elastic net ridge parameter
    lambda2::Float64

    # What happened at each step of the LARS procedure
    steps::Vector{LARSStep}

    # The lambda values at each knot of the LARS procedure. The number
    # of lambdas and coefficient vectors does not necessarily match the
    # number of steps, since a dropped predictor results in an
    # additional knot
    lambdas::Vector{T}

    # Coefficient values at each knot. npredictors x nlambdas
    coefs::Matrix{T}

    # Intercept values at each knot
    intercept::Vector{T}

    LARSPath(method, standardized, lambda2, steps, lambda, coefs) =
        new(method, standardized, lambda2, steps, lambda, coefs)
    LARSPath(method, standardized, lambda2, steps, lambda, coefs, intercept) =
        new(method, standardized, lambda2, steps, lambda, coefs, intercept)
end

function Base.show{T}(io::IO, path::LARSPath{T})
    println(io, "LARSPath{$T} with $(size(path.coefs, 1)) coefficients:")
    !isempty(path.steps) || return

    spacing = repeat(" ", 2)
    steplen = max(iceil(log10(length(path.steps))), 4)
    lambdalen = maximum([length(repr(lambda)) for lambda in path.lambdas])
    print(io, ' ', rpad("Step", steplen), spacing, rpad("λ", lambdalen), spacing, "Action")

    dropped = false
    for i = 1:length(path.steps)
        print(io, '\n', ' ', lpad(repr(i), steplen), spacing,
              rpad(repr(path.lambdas[i]), lambdalen), spacing)

        step = path.steps[i]
        if step.added == 0
            print(io, "- ", join(step.dropped, ","))
        else
            print(io, "+ ", step.added)
        end
    end
end

function swapcols!(X::DenseMatrix, c1::Int, c2::Int)
    for i = 1:size(X, 1)
        @inbounds X[i, c1], X[i, c2] = X[i, c2], X[i, c1]
    end
    X
end

function swaprows!(X::DenseMatrix, c1::Int, c2::Int)
    for i = 1:size(X, 2)
        @inbounds X[c1, i], X[c2, i] = X[c2, i], X[c1, i]
    end
    X
end

@eval function choldelete!{T}(R::StridedView{T}, row::Int)
    inc = stride(R, 2)*sizeof(T)
    p = (row-1)*inc
    for i = row:size(R, 2)-1
        BLAS.blascopy!(i+1, pointer(R)+p+inc, 1, pointer(R)+p, 1)
        p += inc
    end
    for i = row:size(R, 2)-1
        A_mul_B!($(VERSION >= v"0.4.0-dev+2272" ? :(givens(R, i, i+1, i)[1]) : :(givens(R, i, i+1, i))), R)
    end
    return R
end

function standardize!{T}(X::AbstractMatrix{T})
    Xnorm = [begin
        v = view(X, :, i)
        x = 1/sqrt(dot(v, v))
        ifelse(isfinite(x), x, one(T))
    end for i = 1:size(X, 2)]
    scale!(X, Xnorm)
    Xnorm
end

function lars{T<:BlasReal}(X::Matrix{T}, y::Vector{T}; method::Symbol=:lasso,
                           intercept::Bool=true, standardize::Bool=true,
                           lambda2::Float64=0.0, maxiter::Int=typemax(Int),
                           lambda_min::Float64=0.0, use_gram::Bool=(size(X, 1) > size(X, 2)),
                           verbose::Bool=false)
    # Center and standardize
    if intercept
        μX = mean(X, 1)
        X = X .- μX
        μy = mean(y)
        y = y .- μy
    elseif !use_gram || standardize
        X = copy(X)
    end

    if standardize
        Xnorm = standardize!(X)
    end

    nfeatures = size(X, 2)
    nsamples = size(X, 1)
    maxfeatures = min(maxiter, nfeatures)

    coef = zeros(T, nfeatures)
    prev_coef = zeros(T, nfeatures)
    coefs = zeros(T, nfeatures, maxfeatures + 1)
    lambdas = zeros(T, maxfeatures + 1)
    lambda = convert(T, Inf)

    x1 = 1 / (1 + lambda2)
    x2 = 1 / sqrt(1 + lambda2)

    niter, nactive = 1, 0
    # We swap columns of X as the algorithm progresses
    active, indices = Int[], [1:nfeatures;]
    steps = LARSStep[]

    # Holds the sign of covariance of active features
    signactive = Int8[]
    sizehint!(signactive, maxfeatures)

    # Indices of dropped features in the active set
    idx = Int[]

    drop = false

    R = similar(X, maxfeatures, maxfeatures)

    least_squares_buffer = similar(X, maxfeatures)
    eq_dir = similar(y)
    corr_eq_dir_buffer = similar(X, nfeatures)

    Gram = use_gram ? X'X : similar(X, 0, 0)
    Cov = X'y
    scale!(Cov, x2)

    while true
        if isempty(Cov)
            C_idx = 0
            C_ = 0.0
            C = 0.0
        else
            C_idx = 1
            C_ = NaN
            C = NaN
            for i = 1:length(Cov)
                if isnan(C_) || abs(Cov[i]) > C
                    C_idx = i
                    C_ = Cov[i]
                    C = abs(Cov[i])
                end
            end
        end

        prev_lambda = lambda
        lambda = C
        lambdas[niter] = lambda

        if lambda <= lambda_min # early stopping
            if lambda != lambda_min && niter > 1
                # interpolation factor 0 <= ss < 1
                # In the first iteration, all lambdas are zero, the formula
                # below would make ss a NaN
                ss = ((prev_lambda - lambda_min) /
                      (prev_lambda - lambda))
                for i = 1:length(coef)
                    coef[i] = prev_coef[i] + ss * (coef[i] - prev_coef[i])
                end
                lambdas[niter] = lambda_min
                coefs[:, niter] = coef
            end
            break
        end

        if !drop
            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   w )                                   #
            #     R  ->  (       )  , where R' * w = Xa' x_j         #
            #            ( 0   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################
            m, n = nactive+1, C_idx+nactive

            removed_Cov = Cov[C_idx]
            Cov[C_idx] = Cov[1]
            shift!(Cov)
            indices[n], indices[m] = indices[m], indices[n]

            if !use_gram
                swapcols!(X, m, n)
                newcolX = view(X, :, nactive+1)
                R[nactive+1, nactive+1] = dot(newcolX, newcolX)
                Ac_mul_B!(view(R, 1:nactive, nactive+1:nactive+1), view(X, :, 1:nactive), view(X, :, nactive+1:nactive+1))
            else
                swaprows!(Gram, m, n)
                swapcols!(Gram, m, n)
                copy!(view(R, 1:nactive+1, nactive+1), view(Gram, 1:nactive+1, nactive+1))
            end

            if lambda2 != 0.0
                for i = 1:nactive
                    R[i, nactive+1] = R[i, nactive+1] * x1
                end
                R[nactive+1, nactive+1] = (R[nactive+1, nactive+1] + lambda2) * x1
            end

            # Update the cholesky decomposition
            newcolR = view(R, 1:nactive, nactive+1)
            if nactive != 0
                LAPACK.trtrs!('U', 'T', 'N', view(R, 1:nactive, 1:nactive), newcolR)
            end

            v = dot(newcolR, newcolR)
            diag = max(sqrt(abs(R[nactive+1, nactive+1] - v)), eps())
            R[nactive+1, nactive+1] = diag

            if diag < 1e-7
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added
                warn(@sprintf "Regressors in active set degenerate. Dropping a regressor, after %i iterations, i.e. λ=%.3e, with an active set of %i regressors, and the smallest cholesky pivot element being %.3e" niter lambda nactive diag)
                # XXX: need to figure a 'drop for good' way
                unshift!(Cov, removed_Cov)
                Cov[1] = Cov[C_idx]
                Cov[C_idx] = 0
                continue
            end

            nactive += 1
            push!(signactive, sign(C_))
            push!(active, indices[nactive])
            push!(steps, LARSStep(indices[nactive]))

            if verbose
                @printf "%s\t\t%s\t\t%s\t\t%s\t\t%s\n" niter active[end] "" nactive C
            end
        end

        if method == :lasso && niter > 0 && prev_lambda < lambda
            # lambda is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warn(@sprintf "Early stopping the lars path, as the residues are small and the current value of lambda is no longer well controlled. %i iterations, λ=%.3e, previous λ=%.3e, with an active set of %i regressors." niter lambda prev_lambda nactive)
            break
        end

        # least squares solution
        activeR = view(R, 1:nactive, 1:nactive)
        least_squares_buffer[1:nactive] = signactive
        least_squares = LAPACK.potrs!('U', activeR, view(least_squares_buffer, 1:nactive))
        if length(least_squares) == 1 && least_squares[1] == 0
            # This happens because signactive[:nactive] = 0
            least_squares[:] = 1
            AA = one(T)
        else
            AA = one(T) / sqrt(dot(least_squares, signactive))

            # is this really needed ?
            if !isfinite(AA)
                # L is too ill-conditioned
                i = 0
                while !isfinite(AA)
                    x = ((2^i) * eps())
                    for i = 1:nactive
                        activeR[i, i] += x
                    end
                    least_squares = LAPACK.potrs!('U', activeR, least_squares)
                    AA = one(T) / sqrt(max(dot(least_squares, signactive), eps()))
                    i += 1
                end
            end

            scale!(least_squares, AA * x2)
        end

        corr_eq_dir = view(corr_eq_dir_buffer, 1:size(X, 2)-nactive)
        if !use_gram
            # equiangular direction of variables in the active set
            A_mul_B!(eq_dir, view(X, :, 1:nactive), least_squares)
            # correlation between each inactive variables and
            # eqiangular vector
            Ac_mul_B!(corr_eq_dir, view(X, :, nactive+1:size(X, 2)), eq_dir)
        else
            # scikit-learn suggests using QR for this, but doesn't
            Ac_mul_B!(corr_eq_dir, view(Gram, 1:nactive, nactive+1:size(Gram, 2)), least_squares)
        end
        scale!(corr_eq_dir, x2)

        gamma_ = C / AA
        for i = 1:length(Cov)
            p = (C - Cov[i]) / (AA - corr_eq_dir[i])
            gamma_ = ifelse(p > 0 && p < gamma_, p, gamma_)
            p = (C + Cov[i]) / (AA + corr_eq_dir[i])
            gamma_ = ifelse(p > 0 && p < gamma_, p, gamma_)
        end

        drop = false
        z_pos = Inf
        z = -coef[active] ./ least_squares
        for i = 1:length(z)
            if z[i] > 0 && z[i] < z_pos
                z_pos = z[i]
            end
        end
        if z_pos < gamma_
            for i = length(z):-1:1
                if z[i] == z_pos
                    push!(idx, i)
                end
            end

            # update the sign, important for LAR
            signactive[idx] = -signactive[idx]

            if method == :lasso
                gamma_ = z_pos
            end
            drop = true
        end

        niter += 1

        if niter > size(coefs, 2)
            # resize the coefs and lambdas array
            addsteps = 2 * max(1, (maxfeatures - nactive))
            coefs_new = zeros(T, size(coefs, 1), niter+addsteps)
            coefs_new[:, 1:size(coefs, 2)] = coefs
            coefs = coefs_new
            resize!(lambdas, niter+addsteps)
            lambdas[niter:end] = 0
        end
        copy!(prev_coef, coef)
        fill!(coef, zero(eltype(coef)))
        for i = 1:length(active)
            c = prev_coef[active[i]] + gamma_ * least_squares[i]
            coef[active[i]] = c
            coefs[active[i], niter] = c
        end

        niter <= maxiter || break
        nactive - ifelse(method == :lasso, length(idx)+1, 0) < maxfeatures || break

        # update correlations
        for i = 1:length(Cov)
            Cov[i] -= gamma_ * corr_eq_dir[i]
        end

        # See if any coefficient has changed sign
        if drop
            if method == :lasso
                # handle the case when idx is not length of 1
                dropidx = Int[]
                for i = sort(idx, rev=true)
                    choldelete!(activeR, i)
                    push!(dropidx, splice!(active, i))
                    splice!(signactive, i)
                end
                nactive -= length(idx)
                dropidx = tuple(dropidx...)
                push!(steps, LARSStep(dropidx))

                # propagate dropped variable
                if !use_gram
                    for ii in idx
                        for i in ii:nactive
                            indices[i], indices[i + 1] = indices[i + 1], indices[i]
                            swapcols!(X, i, i+1)
                        end
                    end

                    active_coef = view(least_squares_buffer, 1:nactive)
                    for i = 1:nactive
                        active_coef[i] = coef[active[i]]
                    end
                    residual = A_mul_B!(eq_dir, view(X, :, 1:nactive), active_coef)
                    broadcast!(-, residual, y, residual)
                    prepend!(Cov, scale!(view(X, :, nactive+(1:length(idx)))'residual, x2))
                else
                    for ii in idx
                        for i in ii:nactive
                            indices[i], indices[i + 1] = indices[i + 1], indices[i]
                            swaprows!(Gram, i, i+1)
                            swapcols!(Gram, i, i+1)
                        end
                    end

                    residual = eq_dir
                    fill!(residual, zero(T))
                    for i in active
                        BLAS.axpy!(coef[i], view(X, :, i), residual)
                    end
                    broadcast!(-, residual, y, residual)
                    prepend!(Cov, [dot(view(X, :, i), residual) * x2 for i in dropidx])
                end

                if verbose
                   @printf "%s\t\t%s\t\t%s\t\t%s\t\t%s\n" niter "" repr(dropidx) nactive abs(Cov[1])
                end
            end
            empty!(idx)
        end
    end

    coefs = coefs[:, 1:niter]
    lambdas = lambdas[1:niter]
    if standardize
        scale!(Xnorm, coefs)
    end
    if intercept
        LARSPath{T}(method, standardize, lambda2, steps, lambdas, coefs, μy .- vec(μX * coefs))
    else
        LARSPath{T}(method, standardize, lambda2, steps, lambdas, coefs)
    end
end
