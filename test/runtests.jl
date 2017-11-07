using GeoStats
using LocallyWeightedRegression
using Plots; gr()
using Base.Test
using VisualRegressionTests

# setup GR backend for Travis CI
ENV["GKSwstype"] = "100"

# list of maintainers
maintainers = ["juliohm"]

# environment settings
ismaintainer = "USER" ∈ keys(ENV) && ENV["USER"] ∈ maintainers
istravislinux = "TRAVIS" ∈ keys(ENV) && ENV["TRAVIS_OS_NAME"] == "linux"
datadir = joinpath(@__DIR__,"data")

@testset "1D regression problem" begin
  srand(2017)

  N = 100
  x = linspace(0,1, N)
  y = x.^2 .+ [i/1000*randn() for i=1:N]

  geodata = GeoDataFrame(DataFrames.DataFrame(features=x, response=y), [:features])
  domain = bounding_grid(geodata, [N])
  problem = EstimationProblem(geodata, domain, :response)

  solver = LocalWeightRegress(:response => @NT(kernel=ExponentialKernel(10.)))

  solution = solve(problem, solver)

  results = digest(solution)
  yhat = results[:response][:mean]
  yvar = results[:response][:variance]

  if ismaintainer || istravislinux
    function plot_solution(fname)
      scatter(x, y, label="data", size=(1000,400))
      plot!(x, yhat, ribbon=yvar, fillalpha=.5, label="LWR")
      png(fname)
    end
    refimg = joinpath(datadir,"solution.png")

    @test test_images(VisualTest(plot_solution, refimg), popup=!istravislinux) |> success
  end
end
