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

* `weightfun` - Weighting function (default to `exp(-h^2/2)`)
* `distance`  - A distance from Distances.jl (default to `Euclidean()`)
* `neighbors` - Number of neighbors (default to 20% of the data)

### References

Cleveland 1979. *Robust locally weighted regression and smoothing scatterplots.*
"""
@estimsolver LocalWeightRegress begin
  @param weightfun = h -> exp(-3*h^2)
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

      # retrieve non-missing data
      locs = findall(!ismissing, pdata[var])
      𝒟 = view(pdata, locs)
      X = coordinates(𝒟)
      z = 𝒟[var]

      # number of data points for variable
      ndata = length(z)

      # weight function
      w = varparams.weightfun

      # number of nearest neighbors
      k = isnothing(varparams.neighbors) ? ceil(Int, 0.2ndata) : varparams.neighbors

      @assert 0 < k ≤ ndata "invalid number of neighbors"

      # fit search tree
      M = varparams.distance
      if M isa NearestNeighbors.MinkowskiMetric
        tree = KDTree(X, M)
      else
        tree = BruteTree(X, M)
      end

      # pre-allocate memory for results
      varμ = Vector{V}(undef, nelms(pdomain))
      varσ = Vector{V}(undef, nelms(pdomain))

      # pre-allocate memory for coordinates
      x = MVector{ncoords(pdomain),coordtype(pdomain)}(undef)

      # estimation loop
      for location in traverse(pdomain, LinearPath())
        coordinates!(x, pdomain, location)

        # find neighbors
        inds, dists = knn(tree, x, k)
        δs = dists ./ maximum(dists)

        # weighted least-squares
        Wₗ = Diagonal(w.(δs))
        Xₗ = [ones(eltype(X), k) X[:,inds]']
        zₗ = view(z, inds)
        θₗ = Xₗ'*Wₗ*Xₗ \ Xₗ'*Wₗ*zₗ

        # linear combination of response values
        xₒ = [one(eltype(x)); x]
        ẑₒ = θₗ ⋅ xₒ
        rₗ = Wₗ*Xₗ*(Xₗ'*Wₗ*Xₗ\xₒ)
        r̂ₒ = norm(rₗ)

        varμ[location] = ẑₒ
        varσ[location] = r̂ₒ
      end

      push!(μs, var => varμ)
      push!(σs, var => varσ)
    end
  end

  EstimationSolution(pdomain, Dict(μs), Dict(σs))
end

end
