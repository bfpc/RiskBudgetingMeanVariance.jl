using Test
using RiskBudgetingMeanVariance
using LinearAlgebra

const RBMV = RiskBudgetingMeanVariance

const ECOS_FAILURE_RETURNS = [
    0.16000572506386948,
    0.016081789069735807,
    0.11542468543574563,
    0.058707578340858185,
    0.09893007032167594,
]

const ECOS_FAILURE_COVARIANCE = [
     0.03224835599967817  -0.010742703577146918   0.006218276496664593   0.01621431174103683   -0.0023929952850974177
    -0.010742703577146918  0.02309163218758832   -0.009767837694466184  -0.012052611244477358  -0.01649472205271113
     0.006218276496664593 -0.009767837694466184   0.02611557497423503   -0.005050101712491409   0.009810397434355755
     0.01621431174103683  -0.012052611244477358  -0.005050101712491409   0.017096633809603138   0.0033002394989673097
    -0.0023929952850974177 -0.01649472205271113    0.009810397434355755   0.0033002394989673097  0.018754344456747413
]

const ECOS_FAILURE_TARGET_RETURN = 0.14597057398037097
const ECOS_FAILURE_MAX_VOL = 0.20

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
    rb_risk_contributions = RiskBudgetingMeanVariance.risk_contributions(Covs, w_rb)
    rb_risk_contributions = rb_risk_contributions ./ sum(rb_risk_contributions)
    @test rb_risk_contributions ≈ B ./ sum(B) atol=1e-6

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

function test_equivalent_jump_convex()
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

    w_rb = rb_ws(-rets, Covs, B)
    rb_ret = w_rb' * rets

    # Interpolating RB and MV
    # The first and last have small feasible sets, so the errors are a bit larger
    for j in 1:19
        print("$j, ")
        target_ret = rb_ret + j/20*(mark_ret - rb_ret)
        w_rb_i  = RiskBudgetingMeanVariance.rb_ws_cvx(-rets, Covs, B; min_ret=target_ret, max_vol=target_vol)
        w_rb_ii = RiskBudgetingMeanVariance.rb_ws_jump(-rets, Covs, B; min_ret=target_ret, max_vol=target_vol)
        @test w_rb_i ≈ w_rb_ii atol=5e-5
    end
    println()
end

function test_rb_ecos_status()
    n_assets = length(ECOS_FAILURE_RETURNS)
    budgets = ones(n_assets)

    # This portfolio proves that the return and volatility limits are feasible.
    feasible_weights = fill(0.2 / n_assets, n_assets)
    feasible_weights[argmax(ECOS_FAILURE_RETURNS)] += 0.8
    @test dot(ECOS_FAILURE_RETURNS, feasible_weights) >=
        ECOS_FAILURE_TARGET_RETURN - 1e-12
    @test sqrt(dot(
        feasible_weights, ECOS_FAILURE_COVARIANCE * feasible_weights
    )) <= ECOS_FAILURE_MAX_VOL

    error = try
        RBMV.rb_ws_cvx(
            -ECOS_FAILURE_RETURNS,
            ECOS_FAILURE_COVARIANCE,
            budgets;
            min_ret=ECOS_FAILURE_TARGET_RETURN,
            max_vol=ECOS_FAILURE_MAX_VOL,
        )
        nothing
    catch err
        err
    end
    @test error isa RBMV.RBSolveError
    if error isa RBMV.RBSolveError
        @test error.backend == "ECOS"
        @test error.termination == RBMV.JuMP.MOI.NUMERICAL_ERROR
        @test error.primal == RBMV.JuMP.MOI.OTHER_RESULT_STATUS
        @test isnothing(error.previous)
    end
end

function test_markowitz()
    # Returns, standard deviation and correlation
    stds = [0.1, 0.2, 0.2]
    rets = [0.01, 0.02, 0.015]
    Corr = [ 1   -0.2  0.1
            -0.2  1   -0.1
            0.1 -0.1  1  ]

    # Useful, calculated, parameters
    dim = length(rets)
    Covs = [ stds[i]*stds[j]*Corr[i,j] for i in 1:dim, j in 1:dim]

    # Using risk-aversion parameter
    lambdas = [0.1, 0.3, 1.0, 3.0, 10.0]
    ws_markowitz = mmv_lambda(rets, Covs, lambdas; positive=true)
    returns_markowitz = [w' * rets for w in ws_markowitz]
    volatilities_markowitz = [sqrt(w' * Covs * w) for w in ws_markowitz]

    # Basic sanity check
    for i in 1:4
        @test returns_markowitz[i] <= returns_markowitz[i+1] - 1e-6
        @test volatilities_markowitz[i] <= volatilities_markowitz[i+1] - 1e-6
    end

    # Test too high target return
    target_return = 0.03
    w_mmv = mmv_return(rets, Covs, target_return; positive=true)
    @test isnothing(w_mmv)

    # Test too low target volatility
    target_vol = 0.05
    w_mmv = mmv_vol(rets, Covs, target_vol; positive=true)
    @test isnothing(w_mmv)
end

test_basic()
test_equivalent_jump_convex()
test_rb_ecos_status()
test_markowitz()
