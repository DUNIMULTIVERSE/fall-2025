using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV
cd(@__DIR__)
include("PS6_USMAN_SOURCE.jl")  

@testset "PS6 Rust Model Tests" begin
    
    #::: Q1: Data Loading :::
    @testset "Q1: Data" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        
        @test isa(df_long, DataFrame) && isa(Xstate, Matrix)
        @test size(Xstate, 2) == 20 && nrow(df_long) == size(Xstate, 1) * 20
        @test all([:bus_id, :time, :Y, :Odometer, :Xstate] .∈ Ref(names(df_long)))
        @test all(0 .<= df_long.Y .<= 1) && issorted(df_long, [:bus_id, :time])
        
        println("✓ Data: $(size(Xstate,1)) buses, $(nrow(df_long)) obs")
    end
    
    #::: Q2: Flexible Logit :::
    @testset "Q2: Logit" begin
        Random.seed!(123)
        df = DataFrame(Y = rand(0:1, 80), Odometer = rand(50000:300000, 80),
                      RouteUsage = rand([0.2, 0.6], 80), Branded = rand(0:1, 80), time = rand(1:20, 80))
        
        model = estimate_flexible_logit(df)
        @test isa(model, GeneralizedLinearModel) && GLM.converged(model)
        @test all(0 .<= predict(model, df) .<= 1)
        
        println("✓ Logit: converged with $(length(coef(model))) params")
    end
    
    #::: Q3: CCP Components :::
    @testset "Q3: CCP" begin
        Random.seed!(456)
        xbin, zbin, T, β = 3, 2, 5, 0.9
        xval, zval = [100.0, 150.0, 200.0], [0.3, 0.7]
        xtran = rand(6, 3)
        
        # 3a: State space
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        @test nrow(state_df) == 6 && all(state_df.Odometer .∈ Ref(xval))
        
        # 3b: Future values
        train = DataFrame(Y = rand(0:1, 40), Odometer = rand(xval, 40),
                         RouteUsage = rand(zval, 40), Branded = rand(0:1, 40), time = rand(1:T, 40))
        model = glm(@formula(Y ~ Odometer + RouteUsage), train, Binomial())
        
        FV = compute_future_values(state_df, model, xtran, xbin, zbin, T, β)
        @test size(FV) == (6, 2, 6) && all(FV[:,:,1] .== 0) && all(FV[:,:,2:T] .<= 0)
        
        # 3c: FVT1 mapping
        N = 3
        df_test = DataFrame(bus_id = repeat(1:N, inner=T), time = repeat(1:T, outer=N))
        fvt1 = compute_fvt1(df_test, FV, xtran, rand(1:xbin, N, T), rand(1:zbin, N), xbin, rand(0:1, N))
        @test length(fvt1) == N * T && all(isfinite, fvt1)
        
        # 3d: Structural params
        df_struct = DataFrame(Y = rand(0:1, 100), Odometer = rand(100000:300000, 100), Branded = rand(0:1, 100))
        theta = estimate_structural_params(df_struct, randn(100) .* 0.1)
        @test isa(theta, GeneralizedLinearModel) && all(isfinite, coef(theta))
        
        println("✓ CCP: state space, FV, FVT1, θ estimation")
    end
    
    #::: Integration :::
    @testset "Integration" begin
        @test_nowarn main()
        @test all([isdefined(Main, f) for f in [:load_and_reshape_data, :estimate_flexible_logit, 
                   :construct_state_space, :compute_future_values, :compute_fvt1, 
                   :estimate_structural_params, :main]])
        println("✓ Integration: all functions exist and main() runs")
    end
    
end

println("\n" * "="^50)
println("ALL TESTS PASSED ✓")
println("="^50)
