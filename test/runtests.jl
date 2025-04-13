using Test
using RiskBudgetingMeanVariance

function test_basic()
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
    @test mmv_min_ret ≈ 0.012903225806451611 atol=1e-6
    @test mmv_min_vol ≈ 0.07615974802782675 atol=1e-6

    # Target volatility
    target_vol = 0.1

    #
    # Markowitz and RP portfolios
    #
    w_mark = mmv_vol(rets, Covs, target_vol; positive=true)
    mark_ret = w_mark' * rets
    mark_vol = sqrt(w_mark' * Covs * w_mark)
    @test mark_ret ≈ 0.01581305854515207 atol=1e-6
    @test mark_vol ≈ target_vol atol=1e-6
    @test sum(w_mark) ≈ 1 atol=1e-6

    w_rb = rb_ws(-rets, Covs, B)
    rb_ret = w_rb' * rets
    rb_vol = sqrt(w_rb' * Covs * w_rb)
    @test rb_ret ≈ 0.014033636065572732 atol=1e-6
    @test rb_vol ≈ 0.08027116500691502 atol=1e-6
    @test sum(w_rb) ≈ 1 atol=1e-6

    # Interpolating RB and MV
    int_curve = []
    cur_vol = rb_vol
    for j in 0:20
        print("$j, ")
        target_ret = rb_ret + j/20*(mark_ret - rb_ret)
        w_rb_i = rb_ws(-rets, Covs, B; min_ret=target_ret, max_vol=target_vol)
        ret_i = rets' * w_rb_i
        @test ret_i >= target_ret - 1e-6
        vol_i = sqrt(w_rb_i' * Covs * w_rb_i)
        @test vol_i >= cur_vol - 1e-6
        cur_vol = vol_i
        push!(int_curve, (vol_i,ret_i))
    end
    println()

    # Markowitz efficient frontier
    cur_vol = mmv_min_vol
    ret_curve = mmv_min_ret:0.0005:max_ret
    for ret in ret_curve
        w_mmv = mmv_return(rets, Covs, ret; positive=true)
        mmv_std = sqrt( w_mmv' * Covs * w_mmv )

        @test w_mmv' * rets >= ret - 1e-6
        @test mmv_std >= cur_vol - 1e-6
        cur_vol = mmv_std
    end

end

test_basic()