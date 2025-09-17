using Test, Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff

cd(@__DIR__)

# Read in the source code
include("PS2USMAN.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 7
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# test_allwrap.jl
using Test
using Random
using LinearAlgebra
using Statistics

using Optim
using DataFrames
using GLM   # for comparison in logit test
# FreqTables, CSV, HTTP not needed here because we simulate data for tests

# ------------------------------
# Helpers / functions (from user's code)
# ------------------------------

# Question 1 polynomial & minusf
f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
minusf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2

# Question 2 OLS SSR
function ols_ssr(beta, X, y)
    ssr = (y .- X * beta)' * (y .- X * beta)
    return ssr[1]   # scalar
end

# Question 3 logit negative log-likelihood
function logit_nll(alpha, X, d)
    η = X * alpha
    pi1 = exp.(η) ./ (1 .+ exp.(η))
    # Use safe logs to avoid domain issues
    eps = 1e-12
    pi1 = clamp.(pi1, eps, 1 - eps)
    ll = -sum((d .== 1) .* log.(pi1) .+ (d .== 0) .* log.(1 .- pi1))
    return ll
end

# Question 5 multinomial logit negative log-likelihood
function mlogit_nll(alpha, X, y)
    N = length(y)
    K = size(X, 2)
    J = length(unique(y))
    bigY = zeros(N, J)
    for j in 1:J
        bigY[:, j] .= (y .== j)
    end
    bigAlpha = [reshape(alpha, K, J - 1) zeros(K)]
    num = zeros(N, J)
    den = zeros(N)
    for j in 1:J
        num[:, j] = exp.(X * bigAlpha[:, j])
        den .+= num[:, j]
    end
    P = num ./ repeat(den, 1, J)
    eps = 1e-12
    P = clamp.(P, eps, 1 - eps)
    loglike = -sum(bigY .* log.(P))
    return loglike
end

# ------------------------------
# Tests
# ------------------------------

Random.seed!(1234)  # deterministic

@testset "allwrap-like tests" begin

    # ---------- Test 1: polynomial minimization (convergence) ----------
    @testset "polynomial optimization" begin
        res = optimize(minusf, [-7.0], BFGS(); Optim.Options(g_tol=1e-8))
        # optimization should converge
        @test Optim.converged(res) == true
        # minimizer should be finite
        m = Optim.minimizer(res)[1]
        @test isfinite(m)
        # the reported minimum should match function at minimizer (within tol)
        computed_min = Optim.minimum(res)
        direct = minusf([m])
        @test isapprox(computed_min, direct; rtol=1e-6, atol=1e-8)
    end

    # ---------- Test 2: OLS closed-form vs optimizer ----------
    @testset "ols optimizer vs closed-form" begin
        n = 200
        K = 3
        X = hcat(ones(n), randn(n, K-1))
        β_true = [1.2, -0.7, 2.5]
        ϵ = 0.05 * randn(n)
        y = X * β_true .+ ϵ

        # closed-form
        βols = inv(X' * X) * X' * y

        # optimizer (minimize SSR)
        x0 = randn(K)
        res = optimize(b -> ols_ssr(b, X, y), x0, LBFGS(); Optim.Options(g_tol=1e-8))
        @test Optim.converged(res) == true
        β_opt = Optim.minimizer(res)

        # close to closed-form solution
        @test isapprox(β_opt, βols; atol=1e-6, rtol=1e-6)

        # and close to true beta (within noise-based tolerance)
        @test isapprox(βols, β_true; atol=0.05, rtol=0.1)
    end

    # ---------- Test 3: logit optimizer vs GLM ----------
    @testset "logit optimizer vs GLM" begin
        n = 1000
        X = hcat(ones(n), randn(n, 2))
        α_true = [-0.5, 1.25, -0.8]
        η = X * α_true
        p = 1 ./(1 .+ exp.(-η))
        y = rand.(Bernoulli.(p))  # elementwise Bernoulli

        # build DataFrame for GLM comparison
        df = DataFrame(intercept = X[:,1], x1 = X[:,2], x2 = X[:,3], y = y)
        # GLM expects the intercept omitted/handled; we'll use formula with column names
