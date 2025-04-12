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

import Pkg
Pkg.activate("examples")

using Random
using Distributions

using RiskBudgetingMeanVariance

# Parameters
include("parameters.jl")

N   = 252
n_reps = 100
n_reps_int = 4

vol_target = 0.10

function mmv_rb_ret_vol(rets, Covs, vol_target, B)
  ws = mmv_vol(rets, Covs, vol_target; positive=true)
  mmv_ret = rets' * ws
  ws_rb = rb_ws(-rets, Covs, B)
  rb_ret = rets' * ws_rb
  rb_vol = sqrt(ws_rb' * Covs * ws_rb)
  return mmv_ret, vol_target, rb_ret, rb_vol
end

# Different portfolios from estimated parameters

# Simulated returns
# rng = MersenneTwister(2);
rng = MersenneTwister(14);

struct Result
  expected_return::Float64
  volatility::Float64
  weights::Vector{Float64}
end

cases = ["Markowitz 10% vol",
  # "Markowitz λ",
  "Min Vol",
  "RiskParity",
  # "RBMV 10% vol 10% ret",
  "50/50 weigths",
  "RBMV 10% vol 50/50 ret"
]
B = ones(dim) # Risk Parity

results = Dict{String,Vector{Result}}()
for case in cases
  results[case] = Result[]
end

use_true_rets = false
use_true_covs = false
for i in 1:n_reps
  returns = rand(rng, normdist, N)

  means = sum(returns, dims=2)/N
  means = means[:,1]
  if use_true_rets
    means = rets
  end
  errs  = returns .- means
  covs  = (errs * errs') / N
  if use_true_covs
    covs = Covs
  end

  ws = zeros(dim)
  w_mmv = zeros(dim)
  w_rp = zeros(dim)
  for case in cases
    if case == "Markowitz 10% vol"
      ws .= mmv_vol(means, covs, vol_target; positive=true)
      w_mmv .= ws
    elseif case == "Markowitz λ"
      ws .= mmv_lambda(means, covs, [1/2]; positive=true)[1]
    elseif case == "Min Vol"
      ws .= RiskBudgetingMeanVariance.min_vol(means, covs; positive=true)[1]
    elseif case == "RiskParity"
      ws .= rb_ws(-means, covs, B)
      w_rp .= ws
    elseif case == "RBMV 10% vol 10% ret"
      ws = rb_ws(-means, covs, B; min_ret=0.10, max_vol=vol_target)
    # "Derived" portfolios, need to run after all others
    elseif case == "50/50 weigths"
      ws .= 0.5*w_mmv + 0.5*w_rp
    elseif case == "RBMV 10% vol 50/50 ret"
      targ_ret = (0.50*w_mmv' * means + 0.50*w_rp' * means)
      ws = rb_ws(-means, covs, B; min_ret=targ_ret, max_vol=vol_target)
    end
    ret = rets' * ws
    vol = sqrt(ws' * Covs * ws)
    push!(results[case], Result(ret, vol, copy(ws)))
  end
end

# Interpolating RB and MMV
rng = MersenneTwister(13);
rb_mmv_curves = []

for i in 1:n_reps_int
  print("Rep $i: ")
  returns = rand(rng, normdist, N)
  returns = returns[1:dim,:]

  means = sum(returns, dims=2)/N
  means = means[:,1]
  errs  = returns .- means
  covs  = (errs * errs') / N

  w_mmv = mmv_vol(means, covs, vol_target; positive=true)
  w_rb = rb_ws(-means, covs, B)

  ret_mmv = means' * w_mmv
  ret_rb  = means' * w_rb

  int_curve = []
  for j in 0:20
    print("$j, ")
    target_ret = ret_rb + j/20*(ret_mmv - ret_rb)
    w_rb_i = rb_ws(-means, covs, B; min_ret=target_ret, max_vol=0.1)
    ret_i = rets' * w_rb_i
    vol_i = sqrt(w_rb_i' * Covs * w_rb_i)
    push!(int_curve, (vol_i,ret_i))
  end
  push!(rb_mmv_curves, int_curve)
  println()
end

# (True) Markowitz efficient frontier
function efficient_frontier(rets, Covs; positive=true)
  ws_ef = Dict{Float64,Vector{Float64}}()
  risk_ef = Dict{Float64,Float64}()

  _, min_ret, _ = RiskBudgetingMeanVariance.min_vol(rets, Covs; positive)
  ret_curve = min_ret:0.0005:max_ret
  for ret in ret_curve
    w_mmv = mmv_return(rets, Covs, ret; positive=true)
    mmv_std = sqrt( w_mmv' * Covs * w_mmv )

    ws_ef[ret] = w_mmv
    risk_ef[ret] = mmv_std
  end
  vol_curve_ef = [risk_ef[ret] for ret in ret_curve]
  return ret_curve, vol_curve_ef
end

ret_curve, vol_curve_ef = efficient_frontier(rets, Covs; positive=true)

# 1/N
w_1n = ones(dim)/dim
ret_1n = rets' * w_1n
vol_1n = sqrt(w_1n' * Covs * w_1n)

# Graphs
import PyPlot as plt
plt.figure()
plt.plot(vol_curve_ef, ret_curve, label="Efficient Frontier")
for case in cases
  case_rets = [r.expected_return for r in results[case]]
  case_vols = [r.volatility for r in results[case]]
  plt.scatter(case_vols, case_rets, label=case)
end
plt.scatter(vol_1n, ret_1n, color="red", marker="x", label="1/N")
plt.axvline(vol_target, color="black", linestyle="--", label="Target vol")
# plt.scatter(stds[1:dim], rets[1:dim], color="red", marker="x", label="Assets")
plt.legend()
plt.xlabel("Vol")
plt.ylabel("Return")
plt.title("Portfolios from $(n_reps) different estimations")

# Interpolating RB and MMV
(ret_mmv, vol_mmv, ret_rb, vol_rb) = mmv_rb_ret_vol(rets, Covs, vol_target, B)

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6,4), sharex=true, sharey=true)
for (snake, ax) in zip(rb_mmv_curves, axs[:])
  ax.plot(vol_curve_ef, ret_curve)
  ax.plot(first.(snake), last.(snake), ".", markersize=4)
  ax.scatter([vol_rb, vol_mmv], [ret_rb, ret_mmv], marker="x", color="black")
end

plt.savefig("examples/chimney_plot.pdf")

fig, ax = plt.subplots(ncols=2, figsize=(12,4))
for i = 1:n_reps
  rp_res = results["RiskParity"][i]
  ax[1].plot(1:dim, rp_res.weights, alpha=0.1, color="C1")
  mv_res = results["Markowitz 10% vol"][i]
  ax[2].plot(1:dim, mv_res.weights, alpha=0.1, color="C0")
end
ax[1].set_title("Risk Parity")
ax[2].set_title("Markowitz 10% vol")

fig, ax = plt.subplots(ncols=2, figsize=(12,4))
for i = 1:n_reps
  rp_res = results["50/50 weigths"][i]
  ax[1].plot(1:dim, rp_res.weights, alpha=0.1, color="C3")
  mv_res = results["RBMV 10% vol 50/50 ret"][i]
  ax[2].plot(1:dim, mv_res.weights, alpha=0.1, color="C2")
end
ax[1].set_title("50/50 weigths")
ax[2].set_title("RBMV 10% vol 50/50 ret")
ax[1].set_ylim(0, 1)
ax[2].set_ylim(0, 1)

fig, ax = plt.subplots(ncols=1, figsize=(12,4))
for i = 1:n_reps
  rp_res = results["50/50 weigths"][i]
  mv_res = results["RBMV 10% vol 50/50 ret"][i]
  ax.plot(1:dim, rp_res.weights .-  mv_res.weights, alpha=0.1, color="C4")
end
ax.set_title("Difference in weights")

rp_max = zeros(dim)
rp_min = zeros(dim)
mv_max = zeros(dim)
mv_min = zeros(dim)
for i = 1:dim
  rp_max[i] = maximum([r.weights[i] for r in results["RiskParity"]])
  rp_min[i] = minimum([r.weights[i] for r in results["RiskParity"]])
  mv_max[i] = maximum([r.weights[i] for r in results["Markowitz 10% vol"]])
  mv_min[i] = minimum([r.weights[i] for r in results["Markowitz 10% vol"]])
end

nothing
