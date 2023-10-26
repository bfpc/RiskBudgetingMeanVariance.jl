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
import PyPlot as plt

using RiskBudgetingMeanVariance

# Parameters
dim = 5
N   = 100

# Simulated returns out of a N(0.02, I) distribution
rng = MersenneTwister(2);
returns = randn(rng, dim, N) .+ 0.02

means = sum(returns, dims=2)/N
means = means[:,1]
errs  = returns .- means
covs  = (errs * errs') / N

max_ret = maximum(means)

# Markowitz portfolios
ws = Dict{Float64,Vector{Float64}}()
vol_curve = Vector{Float64}()

ret_curve = 0.000:0.0005:max_ret
for ret in ret_curve
    w_mmv = mmv_return(means, covs, ret; positive=true)
    mmv_std = sqrt( w_mmv' * covs * w_mmv )

    ws[ret] = w_mmv
    push!(vol_curve, mmv_std)
end

# Graphs
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13,4))

# w_i vs returns
ws_stack = hcat([ws[ret] for ret in ret_curve]...)
# ax1.stackplot(ret_curve, ws_stack)
ax1.stackplot(vol_curve, ws_stack)
ax1.set_xlabel("Vol")
ax1.set_ylabel("Weights")

ax2.plot(vol_curve, ret_curve)
ax2.set_xlabel("Vol")
ax2.set_ylabel("Return")

nothing
