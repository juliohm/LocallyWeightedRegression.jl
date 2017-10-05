# LocallyWeightedRegression.jl

[![Build Status](https://travis-ci.org/juliohm/LocallyWeightedRegression.jl.svg?branch=master)](https://travis-ci.org/juliohm/LocallyWeightedRegression.jl)
[![CodeCov](https://codecov.io/gh/juliohm/LocallyWeightedRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliohm/LocallyWeightedRegression.jl)

This package provides an implementation of locally weighted regression (a.k.a. LOESS) introduced by [Cleveland 1979](http://www.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf). It is the most natural generalization of [InverseDistanceWeighting.jl](https://github.com/juliohm/InverseDistanceWeighting.jl) in which one is allowed to use a custom weight function instead of distance-based weights.

Like in the inverse distance weighting scheme, this package makes use of k-d trees from the [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) package for fast data lookup. Locally weighted regression (LWR or LOESS) is a popular non-parametric method in machine learning, however it still has poor statistical properties compared to other estimation methods such as Kriging that explicitly model spatial correlation.
