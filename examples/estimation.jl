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
Pkg.activate(".")

using Random
using Distributions

using RiskBudgetingMeanVariance

# Parameters
include("parameters.jl")
N   = 252
n_reps = 100
n_reps_int = 9

# Markowitz and RP portfolios from estimated parameters

# Simulated returns
# rng = MersenneTwister(2);
rng = MersenneTwister(14);

mmv_rets = []
mmv_vols = []

mmv007_rets = []
mmv007_vols = []

min_vol_rets = []
min_vol_vols = []

rb_rets = []
rb_vols = []
B = ones(dim)

ff_rets = []
ff_vols = []

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

  w_mmv = mmv_vol(means, covs, 0.1; positive=true)
  push!(mmv_rets, w_mmv' * rets)
  push!(mmv_vols, sqrt(w_mmv' * Covs * w_mmv))

  w_mmv007 = mmv_vol(means, covs, 0.075; positive=true)
  push!(mmv007_rets, w_mmv007' * rets)
  push!(mmv007_vols, sqrt(w_mmv007' * Covs * w_mmv007))

  w_minvol, _1, _2 = RiskBudgetingMeanVariance.min_vol(means, covs; positive=true)
  push!(min_vol_rets, w_minvol' * rets)
  push!(min_vol_vols, sqrt(w_minvol' * Covs * w_minvol))

  w_rb = rb_ws(-means, covs, B)
  push!(rb_rets, w_rb' * rets)
  push!(rb_vols, sqrt(w_rb' * Covs * w_rb))

  w_ff = 0.5*(w_mmv + w_rb)
  push!(ff_rets, w_ff' * rets)
  push!(ff_vols, sqrt(w_ff' * Covs * w_ff))
end

# Interpolating RB and MMV
rng = MersenneTwister(13);
rb_mmv_curves = []

for i in 1:n_reps_int
  print("Rep $i: ")
  returns = rand(rng, normdist, N)

  means = sum(returns, dims=2)/N
  means = means[:,1]
  errs  = returns .- means
  covs  = (errs * errs') / N

  w_mmv = mmv_vol(means, covs, 0.1; positive=true)
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

# Graphs
import PyPlot as plt
plt.figure()
plt.plot(vol_curve_ef, ret_curve, label="Efficient Frontier")
plt.scatter(mmv_vols, mmv_rets, label="Markowitz simul")
plt.scatter(mmv007_vols, mmv007_rets, label="Markowitz simul 0.07")
plt.scatter(min_vol_vols, min_vol_rets, label="MinVol simul")
plt.scatter(rb_vols, rb_rets, label="RiskParity simul")
plt.scatter(ff_vols, ff_rets, label="50/50 simul")
plt.axvline(0.1, color="black", linestyle="--", label="Target vol")
plt.legend()
plt.xlabel("Vol")
plt.ylabel("Return")
plt.title("$(n_reps) estimated RP and Markowitz portfolios")

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(18,13))
for (snake, ax) in zip(rb_mmv_curves, axs[:])
  ax.plot(vol_curve_ef, ret_curve)
  ax.plot(first.(snake), last.(snake), ".")
  ax.scatter([0.075, 0.10], [0.08, 0.12], marker="x", color="black")
end

nothing
