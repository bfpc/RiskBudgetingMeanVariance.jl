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

# Auxiliary functions
# Concentration / diversification measures
function herfindahl(rc)
  ps = rc / sum(rc)
  return sum(ps.^2)
end

function gini(rc)
  ps = rc / sum(rc)
  sort!(ps)
  n = length(ps)
  return 2*sum(i*ps[i] for i=1:n) / sum(p_i for p_i in ps) / n - (n+1)/n
end

function entropy(rc)
  ps = rc / sum(rc)
  return - sum(log(p_i)*p_i for p_i in ps if p_i > 0)
end


# RiskBudgetingMeanVariance curves

function rb2mmv(means, covs; B=ones(length(means)), target_vol=0.1)
  w_mmv = mmv_vol(means, covs, target_vol; positive=true)
  w_rb = rb_ws(-means, covs, B)

  ret_mmv = means' * w_mmv
  ret_rb  = means' * w_rb

  interpolation_curve = []
  ws = []
  for j in 0:20
    print("$j, ")
    target_ret = ret_rb + j/20*(ret_mmv - ret_rb)
    w_rb_i = rb_ws(-means, covs, B; min_ret=target_ret, max_vol=0.1)
    ret_i = rets' * w_rb_i
    vol_i = sqrt(w_rb_i' * Covs * w_rb_i)
    push!(interpolation_curve, (vol_i,ret_i))
    push!(ws, w_rb_i)
  end
  return interpolation_curve, ws
end

#
# Script
#

import Pkg
Pkg.activate(".")

using Random: MersenneTwister
using RiskBudgetingMeanVariance
import PyPlot as plt

# Parameters
include("parameters.jl")
n_reps_int = 9
N = 252

measure = entropy
title = "Entropy"
# measure = gini
# title = "Gini index"
# measure = herfindahl
# title = "Herfindahl index"

# Interpolating RB and MMV
rng = MersenneTwister(13)
B = ones(dim)

handle_sample = nothing
handle_pop = nothing

plot_contributions = false
if plot_contributions
  fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(18,13), sharey=true)
  fig.suptitle("Evolution of risk contributions for 9 different samples")
end

plt.figure()
for i in 1:n_reps_int
  print("Rep $i: ")
  # Sample returns to estimate means and covariances
  returns = rand(rng, normdist, N)

  means = sum(returns, dims=2)/N
  means = means[:,1]
  errs  = returns .- means
  covs  = (errs * errs') / N

  interpolation_curve, ws = rb2mmv(means, covs; B=B, target_vol=0.1)

  global handle_sample, = plt.plot(0:20, [measure(risk_contributions(Covs, wi)) for wi in ws], color="C0", alpha=0.1, label="Samples")
  plot_contributions && axs[i].stackplot(0:20, 100*hcat([risk_contributions(Covs, wi) for wi in ws]...))

  println()
end

# (True) Markowitz and RB portfolios

begin
  interpolation_curve, ws = rb2mmv(rets, Covs; B=B, target_vol=0.1)

  global handle_pop, = plt.plot(0:20, [measure(risk_contributions(Covs, wi)) for wi in ws], color="C1", label="Population", linewidth=2.5)
  plt.title(title * " of risk contributions")
  plt.legend(handles=[handle_sample, handle_pop])

  if plot_contributions
    plt.figure() # only activate for stackplot
    plt.stackplot(0:20, 100*hcat([risk_contributions(Covs, wi) for wi in ws]...))
    plt.title("Evolution of risk contributions for population parameters")
  end

  println()
end

nothing
