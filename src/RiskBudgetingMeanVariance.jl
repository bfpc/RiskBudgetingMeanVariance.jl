module RiskBudgetingMeanVariance

using LinearAlgebra
using Convex, ECOS
using JuMP, Ipopt
solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

include("markowitz.jl")
include("gauss_rb.jl")

export mmv_lambda, mmv_return, mmv_vol, rb_ws

end # module RiskBudgetingMeanVariance
