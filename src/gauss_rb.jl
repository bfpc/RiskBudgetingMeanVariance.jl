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

const _RB_FEASIBILITY_TOL = 1e-6

function _validated_rb_weights(
  w_rb, means, covs, backend; min_ret=nothing, max_vol=nothing
)
  if isnothing(w_rb) || !all(isfinite, w_rb)
    error("$backend returned non-finite risk-budgeting weights.")
  end

  weight_sum = sum(w_rb)
  if !isfinite(weight_sum) || weight_sum <= 0
    error("$backend returned risk-budgeting weights with invalid sum $weight_sum.")
  end

  weights = vec(w_rb) ./ weight_sum
  if !all(isfinite, weights) || !all(>(0), weights)
    error("$backend returned risk-budgeting weights that are not finite and positive.")
  end
  if !isapprox(sum(weights), 1; atol=_RB_FEASIBILITY_TOL, rtol=0)
    error("$backend returned risk-budgeting weights that do not sum to one.")
  end

  if !isnothing(min_ret)
    portfolio_return = -means' * weights
    if portfolio_return < min_ret - _RB_FEASIBILITY_TOL
      error(
        "$backend returned portfolio return $portfolio_return below " *
        "minimum $min_ret (tolerance $(_RB_FEASIBILITY_TOL))."
      )
    end
  end

  if !isnothing(max_vol)
    portfolio_variance = weights' * covs * weights
    if !isfinite(portfolio_variance) || portfolio_variance < 0
      error("$backend returned invalid portfolio variance $portfolio_variance.")
    end
    portfolio_volatility = sqrt(portfolio_variance)
    if portfolio_volatility > max_vol + _RB_FEASIBILITY_TOL
      error(
        "$backend returned portfolio volatility $portfolio_volatility above " *
        "maximum $max_vol (tolerance $(_RB_FEASIBILITY_TOL))."
      )
    end
  end

  return weights
end

"""
    rb_ws_jump(means, covs, B; min_ret=nothing, max_vol=nothing)

Weights (positive, summing 1) of the interpolating Risk-Budgeting
portfolio corresponding to volatility contributions B_i.
One can also set minimum return and maximum volatility of the resulting
portfolio, in which case it will not be strictly Risk-Budgeting.

means is the mean loss of each asset (so is typically negative),
covs is the covariance matrix of losses.

The budgets in B are relative. Solver status and the returned portfolio are
validated; an unsuccessful solve throws an error.
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
  if !isnothing(min_ret)
    @constraint(m, ret_bound, -mean_loss >= min_ret * sum(m[:w]))
  end
  if !isnothing(max_vol)
    @constraint(m, std_bound, std_loss <= max_vol * sum(m[:w]) )
  end

  optimize!(m)
  if !is_solved_and_feasible(m)
    error(
      "Ipopt failed to solve the risk-budgeting problem: termination status " *
      "$(termination_status(m)), primal status $(primal_status(m))."
    )
  end
  w_rb = value.(w)
  return _validated_rb_weights(
    w_rb, means, covs, "Ipopt"; min_ret, max_vol
  )
end

"""
    rb_ws_cvx(means, covs, B; min_ret=nothing, max_vol=nothing)

Weights (positive, summing 1) of the interpolating Risk-Budgeting
portfolio corresponding to volatility contributions B_i.
One can also set minimum return and maximum volatility of the resulting
portfolio, in which case it will not be strictly Risk-Budgeting.

means is the mean loss of each asset (so is typically negative),
covs is the covariance matrix of losses.

The budgets in B are relative and are normalized internally for numerical
conditioning. Solver status and the returned portfolio are validated; an
unsuccessful solve throws an error.
"""
function rb_ws_cvx(means, covs, B; min_ret=nothing, max_vol=nothing)
  # Aux
  dim = length(means)
  chol_cov = cholesky(covs)
  # B is relative, so unit-sum scaling leaves the model unchanged while
  # improving the conditioning of ECOS's exponential-cone formulation.
  normalized_B = B ./ sum(B)

  # Convex model
  w = Variable(dim, Positive())
  rb_constr = sum(normalized_B[i] * log(w[i]) for i=1:dim) >= 0

  port_loss = means' * w
  port_vol  = norm(chol_cov.U * w)

  constr = Convex.Constraint[rb_constr]
  # Add return / variance constraint
  if !isnothing(min_ret)
    push!(constr, -port_loss >= min_ret * sum(w) )
  end
  if !isnothing(max_vol)
    push!(constr, port_vol <= max_vol * sum(w) )
  end

  pb = minimize(port_vol, constr)

  solve!(pb, ECOS.Optimizer; silent=true)
  termination = Convex.termination_status(pb)
  primal = Convex.primal_status(pb)
  valid_termination = termination in (
    JuMP.MOI.OPTIMAL, JuMP.MOI.ALMOST_OPTIMAL
  )
  valid_primal = primal in (
    JuMP.MOI.FEASIBLE_POINT, JuMP.MOI.NEARLY_FEASIBLE_POINT
  )
  if !valid_termination || !valid_primal
    error(
      "ECOS failed to solve the risk-budgeting problem: termination status " *
      "$termination, primal status $primal."
    )
  end
  return _validated_rb_weights(
    w.value, means, covs, "ECOS"; min_ret, max_vol
  )
end

rb_ws = rb_ws_cvx
