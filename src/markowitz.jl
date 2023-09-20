# Copyright (C) 2023 Bernardo Freitas Paulo da Costa
#
# This file is part of RiskBudgetingMeanVariance.jl.
#
# RiskBudgetingMeanVariance.jl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# RiskBudgetingMeanVariance.jl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# RiskBudgetingMeanVariance.jl. If not, see <https://www.gnu.org/licenses/>.

# Markowitz portfolios

"""
    mmv_lambda(means, covs, lambdas::Vector; positive::Bool=false) -> Vector{Vector}

Returns the weights (summing 1) of the Markowitz mean-variance portfolios
for each risk-aversion parameter `l` in `lambdas`, minimizing
w' * covs * w - l * means' * w.

If `positive`, constrains the weigths to be positive.
"""
function mmv_lambda(means, covs, lambdas::Vector; positive::Bool=false)
  dim = length(means)
  # Base model
  m = Model(solver)
  @variable(m, w[1:dim])
  @constraint(m, sum(w) == 1)
  positive && set_lower_bound.(w, 0)

  # Different compromises
  ws = Vector{Float64}[]
  for l in lambdas
    @objective(m, Min, w' * covs * w - l * means' * w)
    optimize!(m)
    w_mmv = value.(w)
    push!(ws, w_mmv)
  end
  return ws
end

"""
    mmv_return(means, covs, min_return; positive::Bool=false) -> Vector

Returns the weights of the Markowitz mean-variance portfolios with the
target minimum return `min_return`, minimizing the variance.
If `positive`, constrains the weigths to be positive.
"""
function mmv_return(means, covs, min_return; positive::Bool=false)
  dim = length(means)
  # Base model
  m = Model(solver)
  @variable(m, w[1:dim])
  @constraint(m, sum(w) == 1)
  positive && set_lower_bound.(w, 0)

  @constraint(m, means' * w >= min_return)
  @objective(m, Min, w' * covs * w)
  optimize!(m)

  status = primal_status(m)
  if status == FEASIBLE_POINT
    return value.(w)
  end
  if dual_status(m) == INFEASIBILITY_CERTIFICATE
    @warn "Infeasible: try reducing the minimum return."
  else
    @warn "Error solving problem. Primal status is $status"
  end
end

"""
    mmv_vol(means, covs, max_vol; positive::Bool=false) -> Vector

Returns the weights of the Markowitz mean-variance portfolios with the
target maximum volatility `max_vol`, maximizing the returns.
If `positive`, constrains the weigths to be positive.
"""
function mmv_vol(means, covs, max_vol; positive::Bool=false)
  dim = length(means)
  # Base model
  m = Model(solver)
  @variable(m, w[1:dim])
  @constraint(m, sum(w) == 1)
  positive && set_lower_bound.(w, 0)

  @constraint(m, w' * covs * w <= max_vol^2)
  @objective(m, Max, means' * w)
  optimize!(m)

  status = primal_status(m)
  if status == FEASIBLE_POINT
    return value.(w)
  end
  if dual_status(m) == INFEASIBILITY_CERTIFICATE
    @warn "Infeasible: try increasing the maximum volatility."
  else
    @warn "Error solving problem. Primal status is $status"
  end
end
