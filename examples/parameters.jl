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

using Distributions: MvNormal

# Returns and std's for each asset
stds = [0.1, 0.2, 0.15, 0.08, 0.13]
rets = [0.05, 0.12, 0.09, 0.05, 0.15]
dim = length(rets)

# Correlation and covariance matrices
Corr = [1     0.2  0.4  0.25 0.5
        0.2   1   -0.2  0.4  0.6
        0.4  -0.2   1  -0.1  0.3
        0.25  0.4 -0.1  1    0.3
        0.5   0.6  0.3  0.3  1  ]

Covs = [ stds[i]*stds[j]*Corr[i,j] for i in 1:dim, j in 1:dim]
max_ret = maximum(rets)

normdist = MvNormal(rets, Covs)


nothing
