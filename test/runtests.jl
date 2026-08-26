using Test
using RiskBudgetingMeanVariance
using LinearAlgebra
using Random

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
        @test vol_i <= target_vol + 1e-6
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

function synthetic_rb_instance(seed)
    rng = MersenneTwister(seed)
    n_assets = 11
    loadings = randn(rng, n_assets, 3)
    raw_covariance = loadings * transpose(loadings) +
        Diagonal(0.02 .+ 0.08 .* rand(rng, n_assets))
    asset_volatilities = 0.08 .+ 0.10 .* rand(rng, n_assets)
    scaler = Diagonal(asset_volatilities ./ sqrt.(diag(raw_covariance)))
    covariance = scaler * raw_covariance * scaler
    covariance = (covariance + transpose(covariance)) / 2
    expected_returns = 0.01 .+ 0.17 .* rand(rng, n_assets)
    return expected_returns, covariance
end

# A deterministic instance on which ECOS breaks down numerically even though the
# problem is feasible: the target return is a convex combination of the Markowitz
# and risk-parity returns, and both of those respect the volatility ceiling, so
# the same combination of their weights is an explicit feasible point.
#
# The values are shortest-round-trip Float64 literals rather than RNG output, so
# the instance is identical on every Julia version and word size. They were
# produced from MersenneTwister(11) with the generator above; the target return
# is stored as a literal too, since deriving it would make the fixture depend on
# solver output.
const _RB_FIXTURE_RETURNS = [
  0.14963302974311873,
  0.07477708889963876,
  0.16578332306616253,
  0.09017760249465682,
  0.14850421959345064,
  0.06954125251767458,
  0.07261951034672856,
  0.10953709913822252,
  0.162567115919687,
  0.015838572556332836,
  0.04459439927453368,
]

const _RB_FIXTURE_COVARIANCE = [
  0.013039406096525678 0.002283414994200111 -0.0038177202118800727 -0.003326992694442549 0.002493639059796956 0.014561383075154949 0.009380315063160706 -0.019342674454132167 -0.00422119032138117 0.012419224312833953 -0.006416080428538527
  0.002283414994200111 0.018847805894805572 -0.006354629501217959 -0.010031569433120639 -0.022951583601393807 0.0038882528677462542 -0.00817931061802766 -0.0011412121543295226 -0.018998405146921443 0.011490891942024212 0.012642549119861797
  -0.0038177202118800727 -0.006354629501217959 0.03086100541412175 0.019541230822807606 0.0051204673576762515 -0.008869729939329806 0.0020951520229214983 0.013028937325148517 0.0015094531854311362 -0.014841490336951887 0.0034715310844630385
  -0.003326992694442549 -0.010031569433120639 0.019541230822807606 0.015072667849725198 0.010898410715999951 -0.00685218577978042 0.003829763918590136 0.0084224949713582 0.007596822669208027 -0.012905598895658892 -0.0022343941999103884
  0.002493639059796956 -0.022951583601393807 0.0051204673576762515 0.010898410715999951 0.032203188552915755 0.0013236403991056067 0.014726849091082692 -0.007243386325305613 0.023700233652461194 -0.0094057381567832 -0.0198495892226449
  0.014561383075154949 0.0038882528677462542 -0.008869729939329806 -0.00685218577978042 0.0013236403991056067 0.0181566920643969 0.009569991857189018 -0.022863375511765358 -0.005247495620431003 0.015947896427131507 -0.007129947150930337
  0.009380315063160706 -0.00817931061802766 0.0020951520229214983 0.003829763918590136 0.014726849091082692 0.009569991857189018 0.013998459861295873 -0.014791089305662478 0.006742736130319955 0.0033157068215457514 -0.011943590713873176
  -0.019342674454132167 -0.0011412121543295226 0.013028937325148517 0.0084224949713582 -0.007243386325305613 -0.022863375511765358 -0.014791089305662478 0.032024897677564584 0.0022822648858267674 -0.019842781034497277 0.0131776922418984
  -0.00422119032138117 -0.018998405146921443 0.0015094531854311362 0.007596822669208027 0.023700233652461194 -0.005247495620431003 0.006742736130319955 0.0022822648858267674 0.02186622129897705 -0.011948728797586319 -0.013392850568122613
  0.012419224312833953 0.011490891942024212 -0.014841490336951887 -0.012905598895658892 -0.0094057381567832 0.015947896427131507 0.0033157068215457514 -0.019842781034497277 -0.011948728797586319 0.019695958393827588 -0.0007486451178642483
  -0.006416080428538527 0.012642549119861797 0.0034715310844630385 -0.0022343941999103884 -0.0198495892226449 -0.007129947150930337 -0.011943590713873176 0.0131776922418984 -0.013392850568122613 -0.0007486451178642483 0.01566738397244714
]

const _RB_FIXTURE_TARGET_RETURN = 0.15126851471132619
const _RB_FIXTURE_VOL_CEILING = 0.20

function test_rb_fallback_fixture()
    budgets = ones(length(_RB_FIXTURE_RETURNS))
    means = -_RB_FIXTURE_RETURNS

    # ECOS on its own still fails on this instance. If a future ECOS solves it
    # cleanly this assertion fires: regenerate the fixture from a still-failing
    # instance rather than deleting the check, otherwise the test below stops
    # covering the fallback while continuing to pass.
    @test_throws RBSolveError RiskBudgetingMeanVariance.rb_ws_cvx(
        means, _RB_FIXTURE_COVARIANCE, budgets;
        min_ret=_RB_FIXTURE_TARGET_RETURN, max_vol=_RB_FIXTURE_VOL_CEILING,
    )

    # rb_ws recovers by falling back to Ipopt.
    weights = rb_ws(
        means, _RB_FIXTURE_COVARIANCE, budgets;
        min_ret=_RB_FIXTURE_TARGET_RETURN, max_vol=_RB_FIXTURE_VOL_CEILING,
    )
    @test all(isfinite, weights)
    @test all(>(0), weights)
    @test sum(weights) ≈ 1 atol=1e-6
    @test dot(_RB_FIXTURE_RETURNS, weights) >= _RB_FIXTURE_TARGET_RETURN - 1e-6
    @test sqrt(dot(weights, _RB_FIXTURE_COVARIANCE * weights)) <=
        _RB_FIXTURE_VOL_CEILING + 1e-6

    # Budgets are relative, so scaling them must not change the portfolio.
    scaled = rb_ws(
        means, _RB_FIXTURE_COVARIANCE, 1000 .* budgets;
        min_ret=_RB_FIXTURE_TARGET_RETURN, max_vol=_RB_FIXTURE_VOL_CEILING,
    )
    @test scaled ≈ weights atol=1e-7
end

function test_rb_fallback_sweep()
    # Only the invariant is asserted: whichever backend answers, rb_ws returns a
    # valid portfolio. Which seeds break ECOS is deliberately not asserted -- if
    # the MersenneTwister stream or the solver numerics change, this simply
    # exercises different instances and the invariant must still hold.
    volatility_ceiling = 0.20
    for seed in 1:60
        expected_returns, covariance = synthetic_rb_instance(seed)
        budgets = ones(length(expected_returns))

        markowitz = mmv_vol(
            expected_returns, covariance, volatility_ceiling; positive=true
        )
        isnothing(markowitz) && continue
        risk_parity = rb_ws(-expected_returns, covariance, budgets)
        target_return = 0.8 * dot(expected_returns, markowitz) +
            0.2 * dot(expected_returns, risk_parity)

        weights = rb_ws(
            -expected_returns, covariance, budgets;
            min_ret=target_return, max_vol=volatility_ceiling,
        )
        @test all(isfinite, weights)
        @test all(>(0), weights)
        @test sum(weights) ≈ 1 atol=1e-6
        @test dot(expected_returns, weights) >= target_return - 1e-6
        @test sqrt(dot(weights, covariance * weights)) <=
            volatility_ceiling + 1e-6
    end
end

function test_rb_failure_policy()
    budgets = ones(length(_RB_FIXTURE_RETURNS))
    means = -_RB_FIXTURE_RETURNS
    infeasible_return = maximum(_RB_FIXTURE_RETURNS) + 0.01

    # Both backends fail, and the reported error carries the diagnostics of each.
    combined = try
        rb_ws(means, _RB_FIXTURE_COVARIANCE, budgets; min_ret=infeasible_return)
        nothing
    catch err
        err
    end
    @test combined isa RBSolveError
    if combined isa RBSolveError
        @test combined.backend == "Ipopt"
        @test combined.previous isa RBSolveError
        @test combined.previous.backend == "ECOS"
        message = sprint(showerror, combined)
        @test occursin("ECOS", message)
        @test occursin("Ipopt", message)
        @test occursin("termination status", message)
        @test occursin("primal status", message)
    end

    # Each backend also reports on its own when called directly.
    ipopt_error = try
        RiskBudgetingMeanVariance.rb_ws_jump(
            means, _RB_FIXTURE_COVARIANCE, budgets; min_ret=infeasible_return,
        )
        nothing
    catch err
        err
    end
    @test ipopt_error isa RBSolveError
    if ipopt_error isa RBSolveError
        @test ipopt_error.backend == "Ipopt"
        @test isnothing(ipopt_error.previous)
    end

    # A covariance matrix that is not positive definite is a malformed input,
    # not a solver failure: it must propagate rather than be retried on Ipopt,
    # whose formulation would not reject it.
    non_psd = [1.0 2.0; 2.0 1.0]
    @test_throws PosDefException rb_ws([-0.1, -0.1], non_psd, ones(2))
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
test_rb_fallback_fixture()
test_rb_fallback_sweep()
test_rb_failure_policy()
test_markowitz()
