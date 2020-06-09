# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

module LocallyWeightedRegression

using GeoStatsBase

using LinearAlgebra
using NearestNeighbors
using StaticArrays
using Distances

import GeoStatsBase: solve

export LocalWeightRegress

"""
    LocalWeightRegress(var₁=>param₁, var₂=>param₂, ...)

Locally weighted regression (LOESS) estimation solver.

## Parameters

* `weightfun` - Weight function (default to `(x < 1) * (1 - x^3)^3`)
* `distance`  - A distance defined in Distances.jl (default to `Euclidean()`)
* `neighbors` - Number of neighbors (default to all data locations)

### References

Cleveland 1979. *Robust Locally Weighted Regression and Smoothing Scatterplots*
"""
@estimsolver LocalWeightRegress begin
  @param weightfun = x -> (x < 1) * (1 - x^3)^3
  @param distance = Euclidean()
  @param neighbors = nothing
end

function solve(problem::EstimationProblem, solver::LocalWeightRegress)
  # retrieve problem info
  pdata = data(problem)
  pdomain = domain(problem)

  # result for each variable
  μs = []; σs = []

  for covars in covariables(problem, solver)
    for var in covars.names
      # get user parameters
      varparams = covars.params[(var,)]

      # get variable type
      V = variables(problem)[var]

      # get valid data for variable
      X, z = valid(pdata, var)

      # number of data points for variable
      ndata = length(z)

      # allocate memory
      varμ = Vector{V}(undef, npoints(pdomain))
      varσ = Vector{V}(undef, npoints(pdomain))

      # weight function
      weightfun = varparams.weightfun

      # number of nearest neighbors
      k = isnothing(varparams.neighbors) ? ndata : varparams.neighbors

      @assert 0 < k ≤ ndata "invalid number of neighbors"

      # fit search tree
      M = varparams.distance
      if M isa NearestNeighbors.MinkowskiMetric
        tree = KDTree(X, M)
      else
        tree = BruteTree(X, M)
      end

      # pre-allocate memory for coordinates
      x = MVector{ndims(pdomain),coordtype(pdomain)}(undef)

      # estimation loop
      for location in traverse(pdomain, LinearPath())
        coordinates!(x, pdomain, location)

        # find neighbors
        inds, dists = knn(tree, x, k)
        δs = dists ./ maximum(dists)

        # weighted least-squares
        Wₗ = Diagonal(weightfun.(δs))
        Xₗ = [ones(eltype(X), k) X[:,inds]']
        zₗ = view(z, inds)
        θₗ = Xₗ'*Wₗ*Xₗ \ Xₗ'*Wₗ*zₗ

        # add intercept term to estimation location
        xₗ = [one(eltype(x)); x]

        # linear combination of response values
        rₗ = Wₗ*Xₗ*(Xₗ'*Wₗ*Xₗ\xₗ)

        varμ[location] = θₗ ⋅ xₗ
        varσ[location] = norm(rₗ)
      end

      push!(μs, var => varμ)
      push!(σs, var => varσ)
    end
  end

  EstimationSolution(pdomain, Dict(μs), Dict(σs))
end

end
