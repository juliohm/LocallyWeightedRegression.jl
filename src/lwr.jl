# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    LocalWeightRegress(var₁=>param₁, var₂=>param₂, ...)

Locally weighted regression (LOESS) estimation solver.

## Parameters

* `neighbors` - Number of neighbors (default to all data locations)
* `kernel`    - A kernel (or weight) function (default to ExponentialKernel())
* `distance`  - A distance defined in Distances.jl (default to Euclidean())

### References

Cleveland 1979. *Robust Locally Weighted Regression and Smoothing Scatterplots*
"""
@estimsolver LocalWeightRegress begin
  @param neighbors = nothing
  @param kernel = ExponentialKernel()
  @param distance = Euclidean()
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

    @assert ndata > 0 "estimation requires data"

    # allocate memory
    varμ = Vector{V}(undef, npoints(pdomain))
    varσ = Vector{V}(undef, npoints(pdomain))

    # fit search tree
    kdtree = KDTree(X, varparams.distance)

    # determine number of nearest neighbors to use
    k = varparams.neighbors == nothing ? ndata : varparams.neighbors

    @assert k ≤ ndata "number of neighbors must be smaller or equal to number of data points"

    # determine kernel (or weight) function
    kern = varparams.kernel

    # pre-allocate memory for coordinates
    coords = MVector{ndims(pdomain),coordtype(pdomain)}(undef)

    # estimation loop
    for location in SimplePath(pdomain)
      coordinates!(coords, pdomain, location)

      idxs, dists = knn(kdtree, coords, k)

      Xₗ = [ones(eltype(X), k) X[:,idxs]']
      zₗ = view(z, idxs)

      Wₗ = Diagonal([kern(coords, X[:,j]) for j in idxs])

      # weighted least-squares
      θₗ = Xₗ'*Wₗ*Xₗ \ Xₗ'*Wₗ*zₗ

      # add intercept term to estimation location
      xₗ = [one(eltype(coords)), coords...]

      # linear combination of response values
      rₗ = Wₗ*Xₗ*(Xₗ'*Wₗ*Xₗ\xₗ)

      varμ[location] = θₗ ⋅ xₗ
      varσ[location] = norm(rₗ)
    end

    push!(μs, var => varμ)
    push!(σs, var => varσ)
  end

  EstimationSolution(pdomain, Dict(μs), Dict(σs))
end
