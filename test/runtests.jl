using GeoStatsBase
using GeoStatsDevTools
using LocallyWeightedRegression
using Plots; gr()
using VisualRegressionTests
using Test, Pkg, Random

# list of maintainers
maintainers = ["juliohm"]

# environment settings
istravis = "TRAVIS" ∈ keys(ENV)
ismaintainer = "USER" ∈ keys(ENV) && ENV["USER"] ∈ maintainers
datadir = joinpath(@__DIR__,"data")

if ismaintainer
  Pkg.add("Gtk")
  using Gtk
end

@testset "1D regression problem" begin
  Random.seed!(2017)

  N = 100
  x = range(0, stop=1, length=N)
  y = x.^2 .+ [i/1000*randn() for i=1:N]

  geodata = PointSetData(Dict(:y => y), reshape(x, 1, length(x)))
  domain  = bounding_grid(geodata, [N])
  problem = EstimationProblem(geodata, domain, :y)

  solver = LocalWeightRegress(:y => (kernel=ExponentialKernel(10.),))

  solution = solve(problem, solver)

  results = digest(solution)
  yhat = results[:y][:mean]
  yvar = results[:y][:variance]

  if ismaintainer || istravis
    function plot_solution(fname)
      scatter(x, y, label="data", size=(1000,400))
      plot!(x, yhat, ribbon=yvar, fillalpha=.5, label="LWR")
      png(fname)
    end
    refimg = joinpath(datadir,"solution.png")

    @test test_images(VisualTest(plot_solution, refimg), popup=!istravis, tol=0.1) |> success
  end
end
