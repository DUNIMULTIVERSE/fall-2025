using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return df, X, Z, y
end

#---------------------------------------------------
# Multinomial Logit Negative Log-Likelihood
#---------------------------------------------------
function mlogit_with_Z(theta, X, Z, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    alpha = theta[1:end-1]
    gamma = theta[end]
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] .= (y .== j)
    end
    util = zeros(N, J)
    for j = 1:J-1
        util[:, j] .= X * bigAlpha[:, j] .+ gamma .* (Z[:, j] .- Z[:, J])
    end
    util[:, J] .= 0.0
    expU = exp.(util)
    denom = sum(expU, dims=2)
    P = expU ./ denom
    loglike = -sum(bigY .* log.(P .+ 1e-12))
    return loglike
end

function optimize_mlogit(X, Z, y)
    K = size(X, 2)
    J = length(unique(y))
    startvals = [2*rand(K*(J-1)).-1; 0.1]
    result = optimize(theta -> mlogit_with_Z(theta, X, Z, y),
                      startvals, LBFGS(),
                      Optim.Options(g_tol = 1e-5, iterations=10_000, show_trace=false))
    return result.minimizer
end

#---------------------------------------------------
# Multinomial Logit Wrapper for Tests
#---------------------------------------------------
function mlogit(X, Z, y)
    theta_hat = optimize_mlogit(X, Z, y)
    β̂ = theta_hat[1:end-1]
    γ̂ = theta_hat[end]
    return β̂, γ̂
end

#---------------------------------------------------
# Nested Logit Negative Log-Likelihood
#---------------------------------------------------
function nested_logit_with_Z(theta, X, Z, y, nesting_structure)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    betaWC = theta[1:K]
    betaBC = theta[K+1:2K]
    lambdaWC, lambdaBC = theta[2K+1:2K+2]
    gamma = theta[end]
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] .= (y .== j)
    end
    util = zeros(N, J)
    for j in nesting_structure[1]
        util[:, j] .= X * betaWC .+ (gamma/lambdaWC) .* (Z[:, j] .- Z[:, J])
    end
    for j in nesting_structure[2]
        util[:, j] .= X * betaBC .+ (gamma/lambdaBC) .* (Z[:, j] .- Z[:, J])
    end
    util[:, J] .= 0.0
    IV_WC = log.(sum(exp.(util[:, nesting_structure[1]]), dims=2))
    IV_BC = log.(sum(exp.(util[:, nesting_structure[2]]), dims=2))
    num = zeros(N, J)
    for j in nesting_structure[1]
        num[:, j] .= exp.(util[:, j]) .* (exp.(lambdaWC .* IV_WC)).^(1/lambdaWC - 1)
    end
    for j in nesting_structure[2]
        num[:, j] .= exp.(util[:, j]) .* (exp.(lambdaBC .* IV_BC)).^(1/lambdaBC - 1)
    end
    num[:, J] .= 1.0
    dem = exp.(lambdaWC .* IV_WC) .+ exp.(lambdaBC .* IV_BC) .+ 1
    P = num ./ dem
    loglike = -sum(bigY .* log.(P .+ 1e-12))
    return loglike
end

function optimize_nested_logit(X, Z, y, nesting_structure)
    K = size(X, 2)
    startvals = [2*rand(2K).-1; 1.0; 1.0; 0.1]
    result = optimize(theta -> nested_logit_with_Z(theta, X, Z, y, nesting_structure),
                      startvals, LBFGS(),
                      Optim.Options(g_tol = 1e-5, iterations=10_000, show_trace=false))
    return result.minimizer
end

#---------------------------------------------------
# Nested Logit Wrapper for Tests
#---------------------------------------------------
function nested_logit(X, Z, y)
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
    theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    K = size(X, 2)
    βWĈ = theta_hat[1:K]
    βBĈ = theta_hat[K+1:2K]
    λWĈ, λBĈ = theta_hat[2K+1:2K+2]
    γ̂ = theta_hat[end]
    return βWĈ, βBĈ, λWĈ, λBĈ, γ̂
end

#---------------------------------------------------
# Choice Probabilities for Multinomial Logit
#---------------------------------------------------
function choice_probs(x, z, β̂, γ̂)
    K = length(x)
    J = length(z)
    bigAlpha = [reshape(β̂, K, J-1) zeros(K)]
    util = zeros(J)
    for j = 1:J-1
        util[j] = x' * bigAlpha[:, j] + γ̂ * (z[j] - z[J])
    end
    util[J] = 0.0
    expU = exp.(util)
    denom = sum(expU)
    return expU ./ denom
end

#---------------------------------------------------
# Main Function (Optional)
#---------------------------------------------------
function allwrap()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df, X, Z, y = load_data(url)
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))
    println("\n=== MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)
    println("\n=== NESTED LOGIT RESULTS ===")
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
    nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    println("Estimates: ", nlogit_theta_hat)
end

# Uncomment to run
# allwrap()