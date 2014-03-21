using PyCall, PyPlot, LARS
# Benchmark derived from scikit-learn
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

@pyimport numpy as np
@pyimport gc as pygc
@pyimport sklearn.linear_model as sklm
@pyimport sklearn.datasets.samples_generator as sksg

function compute_bench(samples_range, features_range)
    it = 0
    results = [
                "lars_path (with Gram)"=>zeros(length(samples_range), length(features_range)),
                "lars_path (without Gram)"=>zeros(length(samples_range), length(features_range)),
                "LARS.jl (with Gram)"=>zeros(length(samples_range), length(features_range)),
                "LARS.jl (without Gram)"=>zeros(length(samples_range), length(features_range))
              ]
    max_it = length(samples_range) * length(features_range)
    for (i, n_samples) in enumerate(samples_range)
        alpha_min = sqrt(eps())
        lambda_min = alpha_min*n_samples
        for (j, n_features) in enumerate(features_range)
            it += 1
            println("====================")
            @printf "Iteration %03d of %03d\n" it max_it
            println("====================")
            println("n_samples: $n_samples")
            println("n_features: $n_features")

            # To be fair to sklearn, make sure we pass it C arrays
            o = pycall(sksg.pymember("make_regression"), PyObject,
                       n_samples=n_samples, n_features=n_features,
                       n_informative=iceil(n_features / 10),
                       effective_rank=iceil(min(n_samples, n_features) / 10),
                       bias=0.0)
            X = pycall(o["__getitem__"], PyObject, 0)
            y = pycall(o["__getitem__"], PyObject, 1)

            pygc.collect()
            print("benchmarking lars_path (with Gram): ")
            time = @elapsed begin
                G = np.dot(X[:T], X) # precomputed Gram matrix
                Xy = np.dot(X[:T], y)
                niter = length(sklm.lars_path(X, y, Xy=Xy, Gram=G, method="lasso", alpha_min=alpha_min)[1])
            end
            println(time, "s in ", niter)
            results["lars_path (with Gram)"][i, j] = time

            pygc.collect()
            print("benchmarking lars_path (without Gram): ")
            time = @elapsed niter = length(sklm.lars_path(X, y, method="lasso", alpha_min=alpha_min)[1])
            println(time, "s in ", niter)
            results["lars_path (without Gram)"][i, j] = time

            X = o[:__getitem__](0)
            y = o[:__getitem__](1)

            gc()
            print("benchmarking LARS.jl (with Gram): ")
            time = @elapsed length(LARS.lars(X, y, standardize=false, intercept=false, use_gram=true, lambda_min=lambda_min).lambdas)
            println(time, "s in ", niter)
            results["LARS.jl (with Gram)"][i, j] = time

            gc()
            print("benchmarking LARS.jl (without Gram): ")
            @profile time = @elapsed length(LARS.lars(X, y, standardize=false, intercept=false, use_gram=false, lambda_min=lambda_min).lambdas)
            println(time, "s in ", niter)
            results["LARS.jl (without Gram)"][i, j] = time
        end
    end

    return results
end

# Warmup
LARS.lars(rand(10, 5), randn(10))
sklm.lars_path(rand(10, 5), randn(10))

samples_range = 10:500:2010
features_range = 10:500:2010
results = compute_bench(samples_range, features_range)

clf()
max_time = maximum([maximum(t) for t in values(results)])
ax = {}
for (i, label) in enumerate(sort(collect(keys(results))))
    push!(ax, subplot(2, 2, i))
    title(label)
    imshow(results[label]', vmax=max_time, interpolation="none", aspect="auto", origin="lower",
           extent=(samples_range[1]-step(samples_range)/2, samples_range[end]+step(samples_range)/2,
                   features_range[1]-step(features_range)/2, features_range[end]+step(features_range)/2))
    if i % 2 == 1
        ylabel("Features")
    end
    if i >= 3
        xlabel("Samples")
    end
end
colorbar(ax=ax)[:set_label]("Time (Seconds)")
savefig("performance.png")
