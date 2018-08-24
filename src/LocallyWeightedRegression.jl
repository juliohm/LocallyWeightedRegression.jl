# ------------------------------------------------------------------
# Copyright (c) 2017, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

module LocallyWeightedRegression

using GeoStatsBase
using GeoStatsDevTools

using Reexport
using LinearAlgebra
using NearestNeighbors
using StaticArrays
@reexport using Distances

import GeoStatsBase: solve

include("kernels.jl")
include("lwr.jl")

export
  # kernels
  ExponentialKernel,
  GaussianKernel,

  # solver
  LocalWeightRegress

end
