# ------------------------------------------------------------------
# Copyright (c) 2017, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

struct ExponentialKernel{T<:Real}
  α::T
end
(kern::ExponentialKernel)(x, y) = begin
  exp(-kern.α*norm(x-y))
end
ExponentialKernel() = ExponentialKernel(1.)

struct GaussianKernel{T<:Real}
  α::T
end
(kern::GaussianKernel)(x, y) = begin
  exp(-kern.α*norm(x-y)^2)
end
GaussianKernel() = GaussianKernel(1.)
