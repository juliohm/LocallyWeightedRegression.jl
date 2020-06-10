using LocallyWeightedRegression
using GeoStatsBase
using Plots; gr()
using VisualRegressionTests
using Test, Pkg, Random

ENV["GKSwstype"] = "100"

# environment settings
islinux = Sys.islinux()
istravis = "TRAVIS" ∈ keys(ENV)
datadir = joinpath(@__DIR__,"data")
visualtests = !istravis || (istravis && islinux)
if !istravis
  Pkg.add("Gtk")
  using Gtk
end

@testset "LocallyWeightedRegression.jl" begin
  @testset "1D regression" begin
    Random.seed!(2017)

    N = 100
    x = range(0, stop=1, length=N)
    y = x.^2 .+ [i/1000*randn() for i=1:N]

    sdata   = PointSetData(OrderedDict(:y => y), reshape(x, 1, length(x)))
    sdomain = RegularGrid((0.,), (1.,), dims=(N,))
    problem = EstimationProblem(sdata, sdomain, :y)

    solver = LocalWeightRegress(:y => (neighbors=10,))

    solution = solve(problem, solver)

    yhat, yvar = solution[:y]

    if visualtests
      @plottest begin
        scatter(x, y, label="data", size=(1000,400))
        plot!(x, yhat, ribbon=yvar, fillalpha=.5, label="LWR")
      end joinpath(datadir,"solution1D.png") !istravis
    end
  end

  @testset "2D regression" begin
    geodata = PointSetData(OrderedDict(:y => [1.,0.,1.]), [25. 50. 75.;  25. 75. 50.])
    domain  = RegularGrid{Float64}(100,100)
    problem = EstimationProblem(geodata, domain, :y)

    solver = LocalWeightRegress(:y => (neighbors=3,))

    solution = solve(problem, solver)

    if visualtests
      @plottest contourf(solution) joinpath(datadir,"solution2D.png") !istravis
    end
  end
end
