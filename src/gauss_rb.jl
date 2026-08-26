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

"""
    RBSolveError(backend, termination, primal, reason; previous=nothing)

Raised when a risk-budgeting solve does not yield a usable portfolio, either
because the solver reported a status incompatible with returning a solution, or
because the returned point failed validation.

`previous` chains an earlier failure, so that when a fallback solve also fails
the reported error carries the diagnostics of every backend that was tried.
"""
struct RBSolveError <: Exception
  backend::String
  termination::Union{Nothing,JuMP.MOI.TerminationStatusCode}
  primal::Union{Nothing,JuMP.MOI.ResultStatusCode}
  reason::String
  previous::Union{Nothing,RBSolveError}
end

function RBSolveError(backend, termination, primal, reason; previous=nothing)
  return RBSolveError(backend, termination, primal, reason, previous)
end

function Base.showerror(io::IO, e::RBSolveError)
  print(io, "$(e.backend) failed to solve the risk-budgeting problem: $(e.reason)")
  if !isnothing(e.termination)
    print(io, " Solver reported termination status $(e.termination), ")
    print(io, "primal status $(e.primal).")
  end
  if !isnothing(e.previous)
    print(io, "\n  Previously: ")
    showerror(io, e.previous)
  end
  return nothing
end

const _RB_FEASIBILITY_TOL = 1e-6

function _validated_rb_weights(
  w_rb, means, covs, backend; min_ret=nothing, max_vol=nothing,
  termination=nothing, primal=nothing
)
  fail(reason) = throw(RBSolveError(backend, termination, primal, reason))

  if isnothing(w_rb) || !all(isfinite, w_rb)
    fail("the weights are not finite.")
  end

  weight_sum = sum(w_rb)
  if !isfinite(weight_sum) || weight_sum <= 0
    fail("the weights have invalid sum $weight_sum.")
  end

  weights = vec(w_rb) ./ weight_sum
  if !all(isfinite, weights) || !all(>(0), weights)
    fail("the normalized weights are not finite and positive.")
  end
  if !isapprox(sum(weights), 1; atol=_RB_FEASIBILITY_TOL, rtol=0)
    fail("the normalized weights do not sum to one.")
  end

  if !isnothing(min_ret)
    portfolio_return = -means' * weights
    if portfolio_return < min_ret - _RB_FEASIBILITY_TOL
      fail(
        "portfolio return $portfolio_return is below the minimum $min_ret " *
        "(tolerance $(_RB_FEASIBILITY_TOL))."
      )
    end
  end

  if !isnothing(max_vol)
    portfolio_variance = weights' * covs * weights
    if !isfinite(portfolio_variance) || portfolio_variance < 0
      fail("the portfolio variance $portfolio_variance is invalid.")
    end
    portfolio_volatility = sqrt(portfolio_variance)
    if portfolio_volatility > max_vol + _RB_FEASIBILITY_TOL
      fail(
        "portfolio volatility $portfolio_volatility is above the maximum " *
        "$max_vol (tolerance $(_RB_FEASIBILITY_TOL))."
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

The budgets in B are relative and are used as given: unlike the Convex model,
scaling them changes the absolute residual Ipopt tolerates on the risk-budgeting
constraint, and the unscaled form is the tighter one. Solver status and the
returned portfolio are validated; an unsuccessful solve throws an
[`RBSolveError`](@ref).
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
  termination = termination_status(m)
  primal = primal_status(m)
  if !is_solved_and_feasible(m)
    throw(RBSolveError(
      "Ipopt", termination, primal, "the solve was unsuccessful."
    ))
  end
  w_rb = value.(w)
  return _validated_rb_weights(
    w_rb, means, covs, "Ipopt"; min_ret, max_vol, termination, primal
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
unsuccessful solve throws an [`RBSolveError`](@ref).
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
    throw(RBSolveError(
      "ECOS", termination, primal, "the solve was unsuccessful."
    ))
  end
  return _validated_rb_weights(
    w.value, means, covs, "ECOS"; min_ret, max_vol, termination, primal
  )
end

"""
    rb_ws(means, covs, B; min_ret=nothing, max_vol=nothing)

Weights (positive, summing 1) of the interpolating Risk-Budgeting
portfolio corresponding to volatility contributions B_i.
One can also set minimum return and maximum volatility of the resulting
portfolio, in which case it will not be strictly Risk-Budgeting.

means is the mean loss of each asset (so is typically negative),
covs is the covariance matrix of losses.

The budgets in B are relative and are normalized to unit sum before being
handed to a backend, so scaling B does not change the returned portfolio
regardless of which backend answers.

The problem is first solved with Convex.jl and ECOS. ECOS can break down
numerically on feasible instances, so if it does not return a portfolio that
passes validation, the equivalent JuMP/Ipopt model is solved instead. Whichever
backend answers, the returned weights are validated identically: finite,
strictly positive, summing to one, and satisfying `min_ret` and `max_vol` to an
absolute tolerance of $(_RB_FEASIBILITY_TOL).

Throws an [`RBSolveError`](@ref) if neither backend produces a valid portfolio;
errors that are not solver failures, such as a `PosDefException` from a
covariance matrix that is not positive definite, propagate unchanged.
"""
function rb_ws(means, covs, B; min_ret=nothing, max_vol=nothing)
  # The two backends scale the risk-budgeting constraint differently, so
  # normalizing here is what makes the answer independent of the scale of B no
  # matter which one ends up solving.
  normalized_B = B ./ sum(B)
  try
    return rb_ws_cvx(means, covs, normalized_B; min_ret, max_vol)
  catch ecos_err
    # Only retry genuine solver failures: malformed inputs must not be sent
    # into a second solve that might mask them.
    ecos_err isa RBSolveError || rethrow()
    @warn(
      "ECOS did not produce a valid risk-budgeting portfolio; retrying with " *
      "Ipopt. Further occurrences are not reported.",
      termination=ecos_err.termination,
      primal=ecos_err.primal,
      maxlog=1,
    )
    try
      return rb_ws_jump(means, covs, normalized_B; min_ret, max_vol)
    catch ipopt_err
      ipopt_err isa RBSolveError || rethrow()
      throw(RBSolveError(
        ipopt_err.backend, ipopt_err.termination, ipopt_err.primal,
        ipopt_err.reason; previous=ecos_err
      ))
    end
  end
end
