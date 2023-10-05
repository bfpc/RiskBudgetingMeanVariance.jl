# Copyright (C) 2021 - 2023 Bernardo Freitas Paulo da Costa
#
# This file is part of SamplingRB.jl.
#
# SamplingRB.jl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# SamplingRB.jl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# SamplingRB.jl. If not, see <https://www.gnu.org/licenses/>.

import Pkg
Pkg.activate(".")

using Random
using Distributions

using RiskBudgetingMeanVariance

# Parameters
dim = 5
N   = 252
n_reps = 100
n_reps_int = 9

# Covariance matrix
Corr = [1     0.2  0.4  0.25 0.5
        0.2   1   -0.2  0.4  0.6
        0.4  -0.2   1  -0.1  0.3
        0.25  0.4 -0.1  1    0.3
        0.5   0.6  0.3  0.3  1  ]
stds = [0.1, 0.2, 0.15, 0.08, 0.13]
rets = [0.05, 0.12, 0.09, 0.05, 0.15]
Covs = [
        [stds[i]*stds[j]*Corr[i,j] for j in 1:5]
        for i in 1:5
       ]
Covs = hcat(Covs...)
max_ret = maximum(rets)

normdist = MvNormal(rets, Covs)

# Markowitz and RP portfolios from estimated parameters

# Simulated returns
# rng = MersenneTwister(2);
rng = MersenneTwister(14);

mmv_rets = []
mmv_vols = []

rb_rets = []
rb_vols = []
B = ones(dim)

for i in 1:n_reps
  returns = rand(rng, normdist, N)

  means = sum(returns, dims=2)/N
  means = means[:,1]
  errs  = returns .- means
  covs  = (errs * errs') / N

  w_mmv = mmv_vol(means, covs, 0.1; positive=true)
  push!(mmv_rets, w_mmv' * rets)
  push!(mmv_vols, sqrt(w_mmv' * Covs * w_mmv))

  w_rb = rb_ws(-means, covs, B)
  push!(rb_rets, w_rb' * rets)
  push!(rb_vols, sqrt(w_rb' * Covs * w_rb))
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
ws_ef = Dict{Float64,Vector{Float64}}()
risk_ef = Dict{Float64,Float64}()

ret_curve = 0.060:0.0005:max_ret
for ret in ret_curve
    w_mmv = mmv_return(rets, Covs, ret; positive=true)
    mmv_std = sqrt( w_mmv' * Covs * w_mmv )

    ws_ef[ret] = w_mmv
    risk_ef[ret] = mmv_std
end
vol_curve_ef = [risk_ef[ret] for ret in ret_curve]


# Graphs
import PyPlot as plt
plt.figure()
plt.plot(vol_curve_ef, ret_curve)
plt.scatter(mmv_vols, mmv_rets, label="Markowitz simul")
plt.scatter(rb_vols, rb_rets, label="RParity simul")
plt.xlabel("Vol")
plt.ylabel("Return")

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(18,13))
for (snake, ax) in zip(rb_mmv_curves, axs[:])
  ax.plot(vol_curve_ef, ret_curve)
  ax.plot(first.(snake), last.(snake), ".")
end

nothing
