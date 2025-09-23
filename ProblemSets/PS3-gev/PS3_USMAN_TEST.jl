using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
cd(@__DIR__)
include("PS3_USMAN_SOURCE.jl")   # Load your functions

@testset "PS3 USMAN Tests" begin
    # Load dataset
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    # -------------------------
    # Test 1: Multinomial Logit
    # -------------------------
    β̂, γ̂ = mlogit(X, Z, y)
    @testset "Multinomial Logit" begin
        @test length(β̂) > 0                        # β should not be empty
        @test typeof(γ̂) == Float64                 # γ must be a scalar
        @test abs(γ̂) < 10.0                        # sanity bound
    end

    # -------------------------
    # Test 2: Nested Logit
    # -------------------------
    βWĈ, βBĈ, λWĈ, λBĈ, γ̂_nested = nested_logit(X, Z, y)
    @testset "Nested Logit" begin
        @test typeof(βWĈ) <: AbstractVector
        @test typeof(βBĈ) <: AbstractVector
        @test 0.0 < λWĈ <= 1.0        # inclusive value param in (0,1]
        @test 0.0 < λBĈ <= 1.0
        @test abs(γ̂_nested) < 10.0
    end

    # -------------------------
    # Test 3: Choice Probabilities
    # -------------------------
    @testset "Choice Probabilities" begin
        p = choice_probs(X[1, :], Z[1, :], β̂, γ̂)
        @test isapprox(sum(p), 1.0; atol=1e-8)     # probs sum to 1
        @test all(p .>= 0.0)                       # non-negative
    end
end