## Copyright (c) 2017, Júlio Hoffimann Mendes <juliohm@stanford.edu>
##
## Permission to use, copy, modify, and/or distribute this software for any
## purpose with or without fee is hereby granted, provided that the above
## copyright notice and this permission notice appear in all copies.
##
## THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
## WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
## ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
## WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
## ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
## OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
    LocalWeightRegress(var₁=>param₁, var₂=>param₂, ...)

Locally weighted regression (LOESS) estimation solver.

## Parameters

* `neighbors` - Number of neighbors (default to all data locations)
* `kernel`    - A kernel (or weight) function (default to ExponentialKernel())
* `metric`    - A metric defined in Distances.jl (default to Euclidean())

### References

Cleveland 1979. *Robust Locally Weighted Regression and Smoothing Scatterplots*
"""
@estimsolver LocalWeightRegress begin
  @param neighbors = nothing
  @param kernel = ExponentialKernel()
  @param metric = Euclidean()
end

function solve(problem::EstimationProblem, solver::LocalWeightRegress)
  # retrieve problem info
  pdata = data(problem)
  pdomain = domain(problem)

  # result for each variable
  μs = []; σs = []

  for (var,V) in variables(problem)
    # get user parameters
    if var ∈ keys(solver.params)
      varparams = solver.params[var]
    else
      varparams = LocalWeightRegressParam()
    end

    # get valid data for variable
    X, z = valid(pdata, var)

    # number of data points for variable
    ndata = length(z)

    # allocate memory
    varμ = Vector{V}(npoints(pdomain))
    varσ = Vector{V}(npoints(pdomain))

    if ndata > 0
      # fit search tree
      kdtree = KDTree(X, varparams.metric)

      # determine number of nearest neighbors to use
      k = varparams.neighbors == nothing ? ndata : varparams.neighbors

      @assert k ≤ ndata "number of neighbors must be smaller or equal to number of data points"

      # determine kernel (or weight) function
      kern = varparams.kernel

      # estimation loop
      for location in SimplePath(pdomain)
        x = coordinates(pdomain, location)

        idxs, dists = knn(kdtree, x, k)

        Xₗ = [ones(eltype(X), k) X[:,idxs]']
        zₗ = z[idxs]

        Wₗ = diagm([kern(x, X[:,j]) for j in idxs])

        # weighted least-squares
        θₗ = Xₗ'*Wₗ*Xₗ \ Xₗ'*Wₗ*zₗ

        # add intercept term to estimation location
        xₗ = [one(eltype(x)), x...]

        # linear combination of response values
        rₗ = Wₗ*Xₗ*(Xₗ'*Wₗ*Xₗ\xₗ)

        varμ[location] = θₗ ⋅ xₗ
        varσ[location] = rₗ ⋅ rₗ
      end
    end

    push!(μs, var => varμ)
    push!(σs, var => varσ)
  end

  EstimationSolution(pdomain, Dict(μs), Dict(σs))
end
