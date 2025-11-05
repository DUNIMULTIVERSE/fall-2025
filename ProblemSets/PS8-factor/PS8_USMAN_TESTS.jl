using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV, Distributions, LineSearches
cd(@__DIR__)
include("PS8_USMAN_SOURCE.jl")
@testset "PS8 Factor Model Tests" begin

    #==========================================================================
    # Question 1: Data Loading and Base Regression Tests
    ==========================================================================#
    @testset "Question 1: Data Loading and Base Regression" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        
        # Test data loading
        @testset "load_data()" begin
            df = load_data(url)
            @test df isa DataFrame
            @test size(df, 1) > 0  # Has rows
            @test size(df, 2) == 13  # Expected number of columns
            @test "logwage" in names(df)
            @test "black" in names(df)
            @test "hispanic" in names(df)
            @test "female" in names(df)
            @test "schoolt" in names(df)
            @test "gradHS" in names(df)
            @test "grad4yr" in names(df)
            # Check ASVAB columns
            @test "asvabAR" in names(df)
            @test "asvabCS" in names(df)
            @test "asvabMK" in names(df)
            @test "asvabNO" in names(df)
            @test "asvabPC" in names(df)
            @test "asvabWK" in names(df)
        end
    end

    #==========================================================================
    # Question 2: ASVAB Correlations Tests
    ==========================================================================#
    @testset "Question 2: ASVAB Correlations" begin
        # Create test data
        Random.seed!(123)
        n = 100
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "compute_asvab_correlations()" begin
            cordf = compute_asvab_correlations(test_df)
            @test cordf isa DataFrame
            @test size(cordf) == (6, 6)  # 6x6 correlation matrix
            # Diagonal should be 1s (or close to 1)
            @test all(isapprox.(diag(Matrix(cordf)), 1.0, atol=1e-10))
            # Correlation matrix should be symmetric
            cor_mat = Matrix(cordf)
            @test isapprox(cor_mat, cor_mat', atol=1e-10)
            # All correlations should be between -1 and 1
            @test all(-1 .<= cor_mat .<= 1)
        end
    end

    #==========================================================================
    # Question 3: Full Regression Tests
    ==========================================================================#
    @testset "Question 3: Full Regression with ASVAB" begin
        Random.seed!(123)
        n = 100
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "estimate_full_regression()" begin
            result = estimate_full_regression(test_df)
            @test result isa StatsModels.TableRegressionModel
            # Should have 13 coefficients (intercept + 6 demographics + 6 ASVAB)
            @test length(coef(result)) == 13
            @test !any(isnan.(coef(result)))
        end
    end

    #==========================================================================
    # Question 4: PCA Tests
    ==========================================================================#
    @testset "Question 4: PCA Regression" begin
        Random.seed!(123)
        n = 100
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "generate_pca!()" begin
            df_result = generate_pca!(test_df)
            @test df_result isa DataFrame
            @test "asvabPCA" in names(df_result)
            @test length(df_result.asvabPCA) == n
            @test !any(isnan.(df_result.asvabPCA))
            # PCA scores should have mean ≈ 0
            @test isapprox(mean(df_result.asvabPCA), 0.0, atol=1e-10)
        end
    end

    #==========================================================================
    # Question 5: Factor Analysis Tests
    ==========================================================================#
    @testset "Question 5: Factor Analysis Regression" begin
        Random.seed!(123)
        n = 100
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "generate_factor!()" begin
            df_result = generate_factor!(test_df)
            @test df_result isa DataFrame
            @test "asvabFactor" in names(df_result)
            @test length(df_result.asvabFactor) == n
            @test !any(isnan.(df_result.asvabFactor))
        end
    end

    #==========================================================================
    # Question 6: Factor Model MLE Tests
    ==========================================================================#
    @testset "Question 6: Factor Model MLE" begin
        Random.seed!(123)
        n = 50  # Smaller sample for faster tests
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "prepare_factor_matrices()" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            # Check dimensions
            @test size(X) == (n, 7)  # N x K (7 covariates including constant)
            @test length(y) == n
            @test size(Xfac) == (n, 4)  # N x L (4 covariates including constant)
            @test size(asvabs) == (n, 6)  # N x J (6 ASVAB tests)
            
            # Check for NaNs
            @test !any(isnan.(X))
            @test !any(isnan.(y))
            @test !any(isnan.(Xfac))
            @test !any(isnan.(asvabs))
            
            # Check constant columns
            @test all(X[:, end] .== 1.0)
            @test all(Xfac[:, end] .== 1.0)
        end
        
        @testset "factor_model() likelihood computation" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            # Create simple starting values
            L = size(Xfac, 2)  # 4
            J = size(asvabs, 2)  # 6
            K = size(X, 2)  # 7
            
            θ = vcat(
                0.1 * randn(L * J),  # γ parameters
                0.1 * randn(K),       # β parameters
                0.1 * randn(J + 1),   # α parameters
                0.5 * ones(J + 1)     # σ parameters
            )
            
            # Test likelihood computation
            negloglike = factor_model(θ, X, Xfac, asvabs, y, 5)
            
            @test negloglike isa Real
            @test isfinite(negloglike)
            @test negloglike > 0  # Negative log-likelihood should be positive
        end
        
        @testset "run_estimation() convergence" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            # Simple starting values
            L = size(Xfac, 2)
            J = size(asvabs, 2)
            K = size(X, 2)
            
            start_vals = vcat(
                vec(Xfac \ asvabs),   # γ from OLS
                X \ y,                 # β from OLS
                0.1 * randn(J + 1),   # α small random
                0.5 * ones(J + 1)     # σ starting at 0.5
            )
            
            # This test may take time, so we use fewer iterations
            # In practice, you might skip this or use @testset with longer timeout
            θ̂, se, loglike = run_estimation(test_df, start_vals)
            
            # Check outputs
            @test length(θ̂) == length(start_vals)
            @test length(se) == length(θ̂)
            @test all(isfinite.(θ̂))
            @test all(isfinite.(se))
            @test isfinite(loglike)
            @test all(se .> 0)  # Standard errors should be positive
        end
    end

    #==========================================================================
    # Integration Tests
    ==========================================================================#
    @testset "Integration: Full Workflow" begin
        # Test that main() runs without errors (may take time)
        @test_nowarn main()
    end

    #==========================================================================
    # Edge Cases and Error Handling
    ==========================================================================#
    @testset "Edge Cases" begin
        @testset "Empty DataFrame handling" begin
            empty_df = DataFrame()
            @test_throws Exception compute_asvab_correlations(empty_df)
        end
        
        @testset "Missing values handling" begin
            Random.seed!(123)
            n = 50
            df_with_missing = DataFrame(
                logwage = randn(n),
                black = rand(0:1, n),
                hispanic = rand(0:1, n),
                female = rand(0:1, n),
                schoolt = rand(8:16, n),
                gradHS = rand(0:1, n),
                grad4yr = rand(0:1, n),
                asvabAR = [missing; randn(n-1)],
                asvabCS = randn(n),
                asvabMK = randn(n),
                asvabNO = randn(n),
                asvabPC = randn(n),
                asvabWK = randn(n)
            )
            
            # Should handle or error appropriately with missing values
            @test_throws Exception compute_asvab_correlations(df_with_missing)
        end
    end

    #==========================================================================
    # Numerical Stability Tests
    ==========================================================================#
    @testset "Numerical Stability" begin
        @testset "Correlation matrix properties" begin
            Random.seed!(456)
            n = 100
            # Create highly correlated data
            base = randn(n)
            test_df = DataFrame(
                logwage = randn(n),
                black = rand(0:1, n),
                hispanic = rand(0:1, n),
                female = rand(0:1, n),
                schoolt = rand(8:16, n),
                gradHS = rand(0:1, n),
                grad4yr = rand(0:1, n),
                asvabAR = base + 0.1*randn(n),
                asvabCS = base + 0.1*randn(n),
                asvabMK = base + 0.1*randn(n),
                asvabNO = base + 0.1*randn(n),
                asvabPC = base + 0.1*randn(n),
                asvabWK = base + 0.1*randn(n)
            )
            
            cordf = compute_asvab_correlations(test_df)
            cor_mat = Matrix(cordf)
            
            # Check positive definiteness
            eigenvalues = eigvals(cor_mat)
            @test all(eigenvalues .> -1e-10)  # Allow small numerical errors
        end
    end

end

println("\n" * "="^80)
println("All tests completed!")
println("="^80)
