using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
cd(@__DIR__)
include("PS5_USMAN_SOURCE.jl") 
# test_bus_model.jl

using Test
using LinearAlgebra
using Random

# Set seed for reproducible tests
Random.seed!(123)

################################################################################
# TEST SETUP AND UTILITIES
################################################################################

"""
    create_test_data()
Create small synthetic dataset for testing.
"""
function create_test_data()
    # Small test case: 3 buses, 4 time periods, 3 mileage bins, 2 route types
    N, T, xbin, zbin = 3, 4, 3, 2
    
    # Test data
    Y = [0 1 0 1;      # Decisions
          1 0 1 0;
          0 0 1 1]
    
    X = [100 150 200 250;   # Odometer readings
         80  130 180 230;
         120 170 220 270]
    
    Xstate = [1 2 3 2;      # Discretized mileage states (1-3)
              1 1 2 3;
              2 3 3 3]
    
    Zstate = [1, 2, 1]      # Route usage states
    
    B = [0, 1, 0]           # Brand indicators
    
    # Simple transition matrix for testing
    xtran = zeros(zbin*xbin, xbin)
    for z in 1:zbin
        for x in 1:xbin
            row = x + (z-1)*xbin
            if x < xbin
                xtran[row, x+1] = 0.7  # 70% chance mileage increases
                xtran[row, x] = 0.3    # 30% chance stays same
            else
                xtran[row, x] = 1.0    # At max mileage, stay there
            end
        end
    end
    
    # State values
    xval = [100.0, 200.0, 300.0]  # Mileage values
    
    return (
        Y = Y,
        X = X,
        B = B,
        Xstate = Xstate,
        Zstate = Zstate,
        N = N,
        T = T,
        xval = xval,
        xbin = xbin,
        zbin = zbin,
        xtran = xtran,
        β = 0.9
    )
end

################################################################################
# TEST SUITE
################################################################################

@testset "Bus Engine Replacement Model Tests" begin
    
    @testset "Data Loading Functions" begin
        println("Testing data loading...")
        
        @testset "load_static_data" begin
            df_long = load_static_data()
            
            # Basic structure tests
            @test df_long isa DataFrame
            @test ncol(df_long) >= 5  # bus_id, time, Y, Odometer, RouteUsage, Branded
            @test :bus_id in propertynames(df_long)
            @test :time in propertynames(df_long)
            @test :Y in propertynames(df_long)
            @test :Odometer in propertynames(df_long)
            
            # Data integrity tests
            @test all(0 .<= df_long.Y .<= 1)  # Y should be binary
            @test all(df_long.Odometer .>= 0) # Odometer should be non-negative
            @test all(1 .<= df_long.time .<= 20) # Time periods
        end
        
        @testset "load_dynamic_data" begin
            d = load_dynamic_data()
            
            # Test structure
            @test d isa NamedTuple
            @test haskey(d, :Y)
            @test haskey(d, :X)
            @test haskey(d, :B)
            @test haskey(d, :Xstate)
            @test haskey(d, :Zstate)
            
            # Test dimensions
            @test size(d.Y) == size(d.X)
            @test size(d.Y, 1) == length(d.B) == length(d.Zstate)
            @test d.N == size(d.Y, 1)
            @test d.T == size(d.Y, 2)
            
            # Test data ranges
            @test all(0 .<= d.Y .<= 1)  # Binary decisions
            @test all(d.X .>= 0)        # Non-negative odometer
            @test all(0 .<= d.B .<= 1)  # Binary brand
        end
    end
    
    @testset "Static Model Estimation" begin
        println("Testing static model...")
        
        df_long = load_static_data()
        model = estimate_static_model(df_long)
        
        @test model isa StatsModels.TableRegressionModel
        @test coef(model) isa Vector{Float64}
        @test length(coef(model)) == 3  # Intercept, Odometer, Branded
        
        # Test that coefficients are reasonable
        θ_static = coef(model)
        println("Static coefficients: ", θ_static)
        
        # Mileage coefficient should be negative (higher mileage → more likely to replace)
        @test θ_static[2] < 0 "Mileage coefficient should be negative"
    end
    
    @testset "Future Value Computation" begin
        println("Testing future value computation...")
        
        d = create_test_data()
        θ_test = [1.0, -0.1, 0.5]  # Test parameters
        
        # Initialize FV array
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        @testset "FV dimensions" begin
            @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
        end
        
        @testset "Backward recursion" begin
            # Test that function runs without error
            @test_nowarn compute_future_value!(FV, θ_test, d)
            
            # Terminal condition: FV at T+1 should be zero
            @test all(FV[:, :, end] .== 0)
            
            # After computation, FV should have non-zero values for t <= T
            compute_future_value!(FV, θ_test, d)
            @test any(FV[:, :, 1] .!= 0)  # First period should have values
        end
        
        @testset "FV properties" begin
            compute_future_value!(FV, θ_test, d)
            
            # FV should generally decrease with higher mileage (more worn buses have lower value)
            for t in 1:d.T, z in 1:d.zbin, b in 0:1
                fv_slice = FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t]
                # Higher mileage states should have lower or equal future value
                # Note: This might not always hold due to transition dynamics, but generally true
                if length(fv_slice) > 1
                    @test fv_slice[1] >= fv_slice[end] - 1e-10  # Allow small numerical errors
                end
            end
        end
    end
    
    @testset "Log Likelihood Computation" begin
        println("Testing log likelihood...")
        
        d = create_test_data()
        θ_test = [1.0, -0.1, 0.5]
        
        @testset "Likelihood evaluation" begin
            ll_value = log_likelihood_dynamic(θ_test, d)
            
            @test ll_value isa Float64
            @test isfinite(ll_value)
            @test ll_value < 0  # Negative log likelihood should be negative
        end
        
        @testset "Likelihood gradient" begin
            # Test that likelihood changes with parameters
            ll1 = log_likelihood_dynamic(θ_test, d)
            ll2 = log_likelihood_dynamic(θ_test .+ 0.1, d)
            
            @test ll1 != ll2 "Likelihood should change with different parameters"
        end
        
        @testset "Extreme parameter values" begin
            # Test with extreme parameters
            θ_extreme = [100.0, -100.0, 100.0]
            ll_extreme = log_likelihood_dynamic(θ_extreme, d)
            
            @test isfinite(ll_extreme) "Likelihood should handle extreme parameters"
        end
    end
    
    @testset "Optimization Wrapper" begin
        println("Testing optimization...")
        
        d = create_test_data()
        
        @testset "Optimization setup" begin
            # Test with provided starting values
            θ_start = [1.0, -0.1, 0.5]
            
            # Should run without error
            @test_nowarn estimate_dynamic_model(d, θ_start=θ_start)
        end
        
        @testset "Random starting values" begin
            # Test with random starting values
            @test_nowarn estimate_dynamic_model(d, θ_start=nothing)
        end
    end
    
    @testset "Numerical Stability" begin
        println("Testing numerical stability...")
        
        d = create_test_data()
        
        @testset "Large values" begin
            θ_large = [1e6, -1e6, 1e6]
            ll = log_likelihood_dynamic(θ_large, d)
            @test isfinite(ll) "Should handle large parameter values"
        end
        
        @testset "Small values" begin
            θ_small = [1e-6, -1e-6, 1e-6]  
            ll = log_likelihood_dynamic(θ_small, d)
            @test isfinite(ll) "Should handle small parameter values"
        end
        
        @testset "Zero parameters" begin
            θ_zero = [0.0, 0.0, 0.0]
            ll = log_likelihood_dynamic(θ_zero, d)
            @test isfinite(ll) "Should handle zero parameters"
        end
    end
    
    @testset "Integration Tests" begin
        println("Running integration tests...")
        
        @testset "End-to-end pipeline" begin
            # Test that the entire pipeline runs without errors
            @test_nowarn begin
                # Load data
                df_long = load_static_data()
                d_dynamic = load_dynamic_data()
                
                # Estimate static model
                static_model = estimate_static_model(df_long)
                
                # Use static estimates as starting values
                θ_start = coef(static_model)
                
                # Run dynamic estimation
                estimate_dynamic_model(d_dynamic, θ_start=θ_start)
            end
        end
        
        @testset "Main function" begin
            # Test that main function runs
            @test_nowarn main()
        end
    end
end

################################################################################
# PERFORMANCE TESTS
################################################################################

@testset "Performance Tests" begin
    println("Running performance tests...")
    
    d = create_test_data()
    θ_test = [1.0, -0.1, 0.5]
    
    @testset "Future value computation time" begin
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        # Time the future value computation
        elapsed = @elapsed compute_future_value!(FV, θ_test, d)
        
        println("Future value computation time: ", elapsed, " seconds")
        @test elapsed < 1.0  # Should be reasonably fast for test data
    end
    
    @testset "Likelihood computation time" begin
        elapsed = @elapsed log_likelihood_dynamic(θ_test, d)
        
        println("Likelihood computation time: ", elapsed, " seconds")  
        @test elapsed < 2.0  # Should be reasonably fast for test data
    end
end

################################################################################
# RUN ALL TESTS
################################################################################

function run_all_tests()
    println("Starting comprehensive test suite...")
    println("="^60)
    
    @time @testset "Complete Bus Model Test Suite" begin
        # Run all test sets
        include("test_bus_model.jl")
    end
    
    println("="^60)
    println("All tests completed!")
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
