using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

#---------------------------------------------------
# Gauss–Legendre Quadrature (from lgwt.jl)
#---------------------------------------------------
include("ProblemSets/PS4-mixture/lgwt.jl")

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end

#---------------------------------------------------
# Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------
function mlogit_with_Z(theta, X, Z, y)
    K = size(X, 2)
    J = size(Z, 2)
    N = size(X, 1)

    alpha = theta[1:(K*(J-1))]
    gamma = theta[end]

    bigY = zeros(eltype(theta), N, J)
    for j = 1:J
        bigY[:, j] .= y .== j
    end

    bigAlpha = hcat(reshape(alpha, K, J-1), zeros(K,1))

    num = zeros(eltype(theta), N, J)
    for j = 1:J
        num[:,j] .= exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
    end
    dem = sum(num, dims=2)
    P = num ./ dem

    loglike = -sum(bigY .* log.(P .+ 1e-12))
    return loglike
end

#---------------------------------------------------
# Quadrature Practice
#---------------------------------------------------
function practice_quadrature()
    println("=== Question 3a: Quadrature Practice ===")
    d = Normal(0, 1)
    nodes, weights = lgwt(7, -4, 4)
    integral_density = sum(weights .* pdf.(d, nodes))
    expectation = sum(weights .* nodes .* pdf.(d, nodes))
    println("∫φ(x)dx ≈ $integral_density (should be ≈ 1)")
    println("∫xφ(x)dx ≈ $expectation (should be ≈ 0)")
end

#---------------------------------------------------
# Variance with Quadrature
#---------------------------------------------------
function variance_quadrature()
    println("\n=== Question 3b: Variance using Quadrature ===")
    d = Normal(0, 2)
    σ = 2
    nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
    variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
    nodes10, weights10 = lgwt(10, -5*σ, 5*σ)
    variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
    println("Variance with 7 quadrature points: $variance_7pts")
    println("Variance with 10 quadrature points: $variance_10pts")
    println("True variance: $(σ^2)")
end

#---------------------------------------------------
# Monte Carlo Practice
#---------------------------------------------------
function practice_monte_carlo()
    println("\n=== Question 3c: Monte Carlo Integration ===")
    σ = 2
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ

    function mc_integrate(f, a, b, D)
        draws = rand(D) * (b - a) .+ a
        return (b - a) * mean(f.(draws))
    end

    for D in [1000, 1000000]
        println("\nWith D = $D draws:")
        variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)
        density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        println("MC Variance: $variance_mc (true: $(σ^2))")
        println("MC Mean: $mean_mc (true: 0)")
        println("MC Density integral: $density_mc (true: 1)")
    end
end

#---------------------------------------------------
# Mixed Logit (Quadrature)
#---------------------------------------------------
function mixed_logit_quad(theta, X, Z, y, nodes, weights)
    K = size(X, 2)
    J = size(Z, 2)
    N = size(X, 1)

    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]

    bigY = zeros(eltype(theta), N, J)
    for j = 1:J
        bigY[:, j] .= y .== j
    end

    bigAlpha = hcat(reshape(alpha, K, J-1), zeros(K,1))
    P_integrated = zeros(eltype(theta), N, J)

    for r in eachindex(nodes)
        gamma_r = mu_gamma + sigma_gamma * nodes[r]
        num_r = zeros(eltype(theta), N, J)
        for j = 1:J
            num_r[:,j] .= exp.(X * bigAlpha[:,j] .+ gamma_r .* (Z[:,j] .- Z[:,J]))
        end
        dem_r = sum(num_r, dims=2)
        P_r = num_r ./ dem_r
        density_weight = weights[r] * pdf(Normal(0,1), nodes[r])
        P_integrated .+= P_r * density_weight
    end

    loglike = -sum(bigY .* log.(P_integrated .+ 1e-12))
    return loglike
end

#---------------------------------------------------
# Mixed Logit (Monte Carlo)
#---------------------------------------------------
function mixed_logit_mc(theta, X, Z, y, D)
    K = size(X, 2)
    J = size(Z, 2)
    N = size(X, 1)

    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]

    bigY = zeros(eltype(theta), N, J)
    for j = 1:J
        bigY[:, j] .= y .== j
    end

    bigAlpha = hcat(reshape(alpha, K, J-1), zeros(K,1))
    P_integrated = zeros(eltype(theta), N, J)

    gamma_dist = Normal(mu_gamma, sigma_gamma)
    for d = 1:D
        gamma_d = rand(gamma_dist)
        num_d = zeros(eltype(theta), N, J)
        for j = 1:J
            num_d[:,j] .= exp.(X * bigAlpha[:,j] .+ gamma_d .* (Z[:,j] .- Z[:,J]))
        end
        dem_d = sum(num_d, dims=2)
        P_d = num_d ./ dem_d
        P_integrated .+= P_d / D
    end

    loglike = -sum(bigY .* log.(P_integrated .+ 1e-12))
    return loglike
end

#---------------------------------------------------
# Optimization Wrappers
#---------------------------------------------------
function optimize_mlogit(X, Z, y)
    K = size(X, 2)
    J = size(Z, 2)
    startvals = [2*rand(K*(J-1)).-1; 0.1]
    result = optimize(theta -> mlogit_with_Z(theta, X, Z, y),
                     startvals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
                     autodiff = :forward)
    return result.minimizer
end

function optimize_mixed_logit_quad(X, Z, y)
    K = size(X, 2)
    J = size(Z, 2)
    nodes, weights = lgwt(7, -4, 4)
    startvals = [2*rand(K*(J-1)).-1; 0.1; 1.0]
    println("Mixed logit quadrature optimization setup complete (not executed)")
    # Uncomment below to run (may be slow!)
    # result = optimize(theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = :forward)
    # println("Mixed logit quadrature estimates: ", result.minimizer)
    return startvals
end

function optimize_mixed_logit_mc(X, Z, y)
    K = size(X, 2)
    J = size(Z, 2)
    startvals = [2*rand(K*(J-1)).-1; 0.1; 1.0]
    println("Mixed logit Monte Carlo optimization setup complete (not executed)")
    # Uncomment below to run (may be slow!)
    # result = optimize(theta -> mixed_logit_mc(theta, X, Z, y, 1000),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = :forward)
    # println("Mixed logit MC estimates: ", result.minimizer)
    return startvals
end

#---------------------------------------------------
# Main Wrapper
#---------------------------------------------------
function allwrap()
    println("=== Problem Set 4: Multinomial and Mixed Logit ===")
    df, X, Z, y = load_data()
    println("Data loaded successfully! Sample size: $(size(X,1)), K=$(size(X,2)), J=$(size(Z,2))")

    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)

    println("\n=== QUESTION 2: INTERPRETATION ===")
    println("γ̂ is the coefficient on the alternative-specific covariate Z. If γ̂ is close to your PS3 result, it means the effect of Z is similar in both models. If it differs, it suggests the model specification or data structure affects the estimated impact of Z.")

    practice_quadrature()
    variance_quadrature()
    practice_monte_carlo()

    println("\n=== QUESTION 4: MIXED LOGIT QUADRATURE (SETUP) ===")
    optimize_mixed_logit_quad(X, Z, y)

    println("\n=== QUESTION 5: MIXED LOGIT MONTE CARLO (SETUP) ===")
    optimize_mixed_logit_mc(X, Z, y)

    println("\n=== ALL ANALYSES COMPLETE ===")
end

# Run the main analysis
allwrap()