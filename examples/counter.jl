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

"""
We build 2 examples for which the RiskBudgeting interpolation
decreases risk while, at the same time, increasing returns.
It is a simple idea: the RiskBudgeting is extreme in its allocation,
so that deviating from it (to achieve better returns) also equilibrates
the portfolio to have less risk.

The first example is for 3 assets, and the RBI stays inside the (Markowitz)
efficient frontier.  The second example uses only 2 assets, so it starts
below the frontier, and joins it near the minimum volatility portfolio;
doing so, it decreases risk and increases returns.
Notice that, with only 2 assets, minimizing volatility subject to a return
constraint yields the Markowitz portfolio - the RB constraint only serves
as the normalization constraint.
"""

import Pkg
Pkg.activate(".")

using RiskBudgetingMeanVariance

#
# Parameters
#
case = 3

# RB weights
B = [20, 1, 20]

# Returns, standard deviation and correlation
stds = [0.1, 0.2, 0.2]
rets = [0.01, 0.02, 0.015]
Corr = [ 1   -0.2  0.1
        -0.2  1   -0.1
         0.1 -0.1  1  ]

# Truncate if needed
if case == 2
  B = B[1:2]

  stds = stds[1:2]
  rets = rets[1:2]
  Corr = Corr[1:2, 1:2]
end

# Useful, calculated, parameters
dim = length(rets)
Covs = [ stds[i]*stds[j]*Corr[i,j] for i in 1:dim, j in 1:dim]
max_ret = maximum(rets)
_, mmv_min_ret, mmv_min_vol = RiskBudgetingMeanVariance.min_vol(rets, Covs)

target_vol = 0.1

#
# Markowitz and RP portfolios
#
w_mark = mmv_vol(rets, Covs, target_vol; positive=true)
mark_ret = w_mark' * rets
mark_vol = sqrt(w_mark' * Covs * w_mark)

w_rb = rb_ws(-rets, Covs, B)
rb_ret = w_rb' * rets
rb_vol = sqrt(w_rb' * Covs * w_rb)

# Interpolating RB and MMV
int_curve = []
for j in 0:20
  print("$j, ")
  target_ret = rb_ret + j/20*(mark_ret - rb_ret)
  w_rb_i = rb_ws(-rets, Covs, B; min_ret=target_ret, max_vol=target_vol)
  ret_i = rets' * w_rb_i
  vol_i = sqrt(w_rb_i' * Covs * w_rb_i)
  push!(int_curve, (vol_i,ret_i))
end
println()

# (True) Markowitz efficient frontier
ws_ef = Dict{Float64,Vector{Float64}}()
risk_ef = Dict{Float64,Float64}()

ret_curve = mmv_min_ret:0.0005:max_ret
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
plt.plot(vol_curve_ef, ret_curve, label="Efficient frontier")
plt.plot(first.(int_curve), last.(int_curve), label="RB Interpolation")
plt.axhline(mmv_min_ret, color="black", linewidth=1, linestyle="dashed")
plt.legend()

plt.xlabel("Vol")
plt.ylabel("Return")

nothing
