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

#
# Risk Budgeting for Volatility risk measure
#

# Marginal risk and risk contributions
function marginal_risks(covs, w)
  Σw = covs * w
  σ  = sqrt(w' * Σw)

  return Σw/σ
end

function risk_contributions(covs, w)
  return w .* marginal_risks(covs, w)
end

# Auxiliary function to evaluate standard deviation of a portfolio in JuMP
function std_port(cov, w)
  d = length(w)
  sqrt(sum(w[i] * cov[i,j] * w[j] for i=1:d for j=1:d))
end

# TODO: check solution for feasibility
"""
    rb_ws_jump(means, covs, B; min_ret=nothing, max_vol=nothing)

Weights (positive, summing 1) of the interpolating Risk-Budgeting
portfolio corresponding to volatility contributions B_i.
One can also set minimum return and maximum volatility of the resulting
portfolio, in which case it will not be strictly Risk-Budgeting.

means is the mean loss of each asset (so is typically negative),
covs is the covariance matrix of losses.
"""
function rb_ws_jump(means, covs, B; min_ret=nothing, max_vol=nothing)
  # Aux
  dim = length(means)

  # Base model
  m = Model(solver)
  @variable(m, w[1:dim] >= 0)
  @constraint(m, sum(B[i] * log(w[i]) for i=1:dim) >= 0)

  @expression(m, mean_loss, means' * w)
  @expression(m, std_loss, std_port(covs, w))

  @objective(m, Min, std_loss)


  # Add return / variance constraint
  if min_ret != nothing
    @constraint(m, ret_bound, -mean_loss >= min_ret * sum(m[:w]))
  end
  if max_vol != nothing
    @constraint(m, std_bound, std_loss <= max_vol * sum(m[:w]) )
  end

  optimize!(m)
  w_rb = value.(w)
  return w_rb ./ sum(w_rb)
end

"""
    rb_ws_cvx(means, covs, B; min_ret=nothing, max_vol=nothing)

Weights (positive, summing 1) of the interpolating Risk-Budgeting
portfolio corresponding to volatility contributions B_i.
One can also set minimum return and maximum volatility of the resulting
portfolio, in which case it will not be strictly Risk-Budgeting.

means is the mean loss of each asset (so is typically negative),
covs is the covariance matrix of losses.
"""
function rb_ws_cvx(means, covs, B; min_ret=nothing, max_vol=nothing)
  # Aux
  dim = length(means)
  chol_cov = cholesky(covs)

  # Convex model
  w = Variable(dim, Positive())
  rb_constr = sum(B[i] * log(w[i]) for i=1:dim) >= 0

  port_loss = means' * w
  port_vol  = norm(chol_cov.U * w)

  constr = Convex.Constraint[rb_constr]
  # Add return / variance constraint
  if min_ret != nothing
    push!(constr, -port_loss >= min_ret * sum(w) )
  end
  if max_vol != nothing
    push!(constr, port_vol <= max_vol * sum(w) )
  end

  pb = minimize(port_vol, constr)

  solve!(pb, ECOS.Optimizer; silent=true)
  w_rb = w.value[:]
  return w_rb ./ sum(w_rb)
end

rb_ws = rb_ws_cvx


