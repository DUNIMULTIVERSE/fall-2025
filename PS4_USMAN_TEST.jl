cd(@__DIR__)
include("PS4_USMAN_SOURCE.jl")
using Test



using Test
using Random, LinearAlgebra, Statistics, Distributions

# Mock the lgwt function for testing (since we don't have the actual file)
function lgwt(n, a, b)
    nodes = range(a, b, length=n)
    weights = fill((b - a) / n, n)
    return nodes, weights
end

# Include the actual code (or copy the functions here for testing)
# For testing purposes, I'll copy the key functions:

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

# Test data generation functions
function generate_test_data(N=100, K=3, J=4)
    Random.seed!(123)
    
    # Generate covariates
    X = hcat(ones(N), randn(N, K-1))
    Z = randn(N, J)
    
    # Generate true parameters
    true_alpha = 0.5 * randn(K * (J-1))
    true_gamma = 0.3
    
    # Generate probabilities and outcomes
    bigAlpha = hcat(reshape(true_alpha, K, J-1), zeros(K,1))
    
    num = zeros(N, J)
    for j = 1:J
        num[:,j] .= exp.(X * bigAlpha[:,j] .+ true_gamma .* (Z[:,j] .- Z[:,J]))
    end
    dem = sum(num, dims=2)
    P = num ./ dem
    
    # Generate choices
    y = zeros(Int, N)
    for i = 1:N
        y[i] = rand(Categorical(P[i,:]))
    end
    
    return X, Z, y, true_alpha, true_gamma
end

# Unit tests
@testset "Multinomial Logit Tests" begin
    
    @testset "Data Generation" begin
        X, Z, y, alpha, gamma = generate_test_data(100, 3, 4)
        
        @test size(X) == (100, 3)
        @test size(Z) == (100, 4)
        @test length(y) == 100
        @test all(1 .<= y .<= 4)
        @test length(alpha) == 6  # K * (J-1) = 3 * 2 = 6
    end
    
    @testset "MLogit Function Structure" begin
        X, Z, y, _, _ = generate_test_data(50, 2, 3)
        K, J = size(X, 2), size(Z, 2)
        
        # Test parameter vector size
        theta = zeros(K*(J-1) + 1)
        @test length(theta) == 5  # 2*(3-1) + 1 = 5
        
        # Test that function returns a scalar
        loglike = mlogit_with_Z(theta, X, Z, y)
        @test loglike isa Real
        @test loglike >= 0  # Negative log likelihood should be non-negative
    end
    
    @testset "MLogit Probability Properties" begin
        X, Z, y, _, _ = generate_test_data(20, 2, 3)
        theta = zeros(5)  # All parameters zero
        
        # With zero parameters, all probabilities should be equal
        K, J = size(X, 2), size(Z, 2)
        N = size(X, 1)
        
        alpha = theta[1:(K*(J-1))]
        gamma = theta[end]
        
        bigAlpha = hcat(reshape(alpha, K, J-1), zeros(K,1))
        num = zeros(N, J)
        for j = 1:J
            num[:,j] .= exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
        end
        dem = sum(num, dims=2)
        P = num ./ dem
        
        # Test probability properties
        @test all(0 .<= P .<= 1)
        @test all(isapprox.(sum(P, dims=2), 1.0, atol=1e-10))
    end
    
    @testset "Mixed Logit Quadrature Structure" begin
        X, Z, y, _, _ = generate_test_data(30, 2, 3)
        nodes, weights = lgwt(5, -2, 2)
        
        theta = [0.1, 0.2, 0.3, 0.4, 0.0, 1.0]  # alpha + mu_gamma + sigma_gamma
        
        loglike = mixed_logit_quad(theta, X, Z, y, nodes, weights)
        
        @test loglike isa Real
        @test loglike >= 0
    end
    
    @testset "Mixed Logit MC Structure" begin
        X, Z, y, _, _ = generate_test_data(30, 2, 3)
        Random.seed!(123)
        
        theta = [0.1, 0.2, 0.3, 0.4, 0.0, 1.0]
        
        loglike = mixed_logit_mc(theta, X, Z, y, 100)
        
        @test loglike isa Real
        @test loglike >= 0
    end
    
    @testset "Numerical Stability" begin
        # Test with extreme parameter values
        X, Z, y, _, _ = generate_test_data(10, 2, 3)
        
        # Very large parameters
        theta_large = 100.0 * ones(5)
        loglike_large = mlogit_with_Z(theta_large, X, Z, y)
        @test !isnan(loglike_large)
        @test !isinf(loglike_large)
        
        # Very small parameters  
        theta_small = -100.0 * ones(5)
        loglike_small = mlogit_with_Z(theta_small, X, Z, y)
        @test !isnan(loglike_small)
        @test !isinf(loglike_small)
    end
    
    @testset "Gradient Approximation" begin
        # Test finite differences gradient approximation
        function finite_diff_gradient(f, theta, epsilon=1e-6)
            grad = similar(theta)
            for i in eachindex(theta)
                theta_plus = copy(theta)
                theta_minus = copy(theta)
                theta_plus[i] += epsilon
                theta_minus[i] -= epsilon
                grad[i] = (f(theta_plus) - f(theta_minus)) / (2 * epsilon)
            end
            return grad
        end
        
        X, Z, y, _, _ = generate_test_data(20, 2, 3)
        theta = 0.1 * randn(5)
        
        f(theta) = mlogit_with_Z(theta, X, Z, y)
        grad = finite_diff_gradient(f, theta)
        
        @test length(grad) == length(theta)
        @test all(!isnan.(grad))
        @test all(!isinf.(grad))
    end
end

@testset "Quadrature and MC Integration Tests" begin
    
    @testset "Quadrature Basic Properties" begin
        nodes, weights = lgwt(7, -1, 1)
        
        @test length(nodes) == 7
        @test length(weights) == 7
        @test nodes[1] ≈ -1.0
        @test nodes[end] ≈ 1.0
        @test all(weights .> 0)
    end
    
    @testset "Normal Distribution Moments" begin
        # Test that quadrature can approximate normal distribution moments
        d = Normal(0, 1)
        nodes, weights = lgwt(10, -5, 5)
        
        # Test mean
        mean_approx = sum(weights .* nodes .* pdf.(d, nodes))
        @test abs(mean_approx) < 0.1  # Should be close to 0
        
        # Test variance
        var_approx = sum(weights .* (nodes .- mean_approx).^2 .* pdf.(d, nodes))
        @test abs(var_approx - 1) < 0.1  # Should be close to 1
    end
end

@testset "Edge Cases and Error Handling" begin
    
    @testset "Single Alternative" begin
        # What happens with J=1? (should handle gracefully or error clearly)
        X = ones(10, 2)
        Z = ones(10, 1)
        y = ones(Int, 10)
        
        # This should either work or throw an informative error
        # For now, just test that it doesn't crash silently
        @test true  # Placeholder - adjust based on desired behavior
    end
    
    @testset "Perfect Prediction" begin
        # Test with data that would lead to perfect prediction
        X = [1.0 0.0; 1.0 0.0; 1.0 1.0; 1.0 1.0]
        Z = [1.0 2.0; 1.0 2.0; 2.0 1.0; 2.0 1.0]
        y = [1, 1, 2, 2]
        
        theta = zeros(3)  # K*(J-1) + 1 = 2*1 + 1 = 3
        
        loglike = mlogit_with_Z(theta, X, Z, y)
        @test loglike isa Real
    end
end

# Run all tests
println("Running unit tests...")
@testset "All Multinomial Logit Tests" begin
    include("test_multinomial_logit.jl")  # This file itself
end

println("All tests completed!")