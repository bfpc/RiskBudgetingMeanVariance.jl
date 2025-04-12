# RiskBudgetingMeanVariance.jl

This package calculates portfolios interpolating between Risk budgeting and
("Markowitz") Mean-Variance, using a convex optimization problem.

## Install

`import Pkg; Pkg.add("RiskBudgetingMeanVariance")`

## Example

```julia
using RiskBudgetingMeanVariance

#
# Parameters
#

# RB weights
B = [2, 3, 1]

# Returns, standard deviation and correlation
stds = [0.1, 0.2, 0.2]
rets = [0.01, 0.02, 0.015]
Corr = [ 1   -0.2  0.1
        -0.2  1   -0.1
         0.1 -0.1  1  ]

# Useful, calculated, parameters
dim = length(rets)
Covs = [ stds[i]*stds[j]*Corr[i,j] for i in 1:dim, j in 1:dim]
max_ret = maximum(rets)
_, mmv_min_ret, mmv_min_vol = RiskBudgetingMeanVariance.min_vol(rets, Covs)

# Target volatility
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

# Interpolating RB and MV
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

# Markowitz efficient frontier
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


# Graphs for the efficient frontier, RBMV interpolation,
#   and the return for the minimum variance portfolio
import PyPlot as plt
plt.figure()
plt.plot(vol_curve_ef, ret_curve, label="Efficient frontier")
plt.plot(first.(int_curve), last.(int_curve), label="RB Interpolation")
plt.axhline(mmv_min_ret, color="black", linewidth=1, linestyle="dashed")
plt.legend()

plt.xlabel("Vol")
plt.ylabel("Return")

nothing
```
