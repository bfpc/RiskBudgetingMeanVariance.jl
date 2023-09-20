module RiskBudgetingMeanVariance

using JuMP, Ipopt
solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

include("markowitz.jl")

export mmv_lambda, mmv_return, mmv_vol

end # module RiskBudgetingMeanVariance
