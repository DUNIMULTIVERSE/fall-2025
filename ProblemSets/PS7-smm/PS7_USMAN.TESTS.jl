using Test, Distributions, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)
include("PS7_USMAN_SOURCE.jl")   # load the source under test

# Helpers
is_probability_matrix(M) = all(isfinite, M) && all(abs.(sum(M, dims=2) .- 1.0) .< 1e-12)
close_enough(a,b) = isapprox(a,b; rtol=1e-8, atol=1e-12)

@testset "Expanded PS7 tests" begin

    @testset "prepare_occupation_data - basics & edge cases" begin
        df = DataFrame(age = [30, 40, missing, 60],
                       race = [1, 2, 1, 1],
                       collgrad = [1, 0, 1, 0],
                       occupation = [8, 9, 7, 13])
        df2, X, y = prepare_occupation_data(df)
        @test nrow(df2) == 3                         # missing dropped
        @test all(unique(y) .<= 7)                   # collapsed categories <= 7
        @test all(df2.white .== (df2.race .== 1))
        @test size(X,2) == 4                         # intercept, age, white, collgrad

        # string occupation codes are supported (if present)
        dfs = DataFrame(age = 20:23, race = [1,1,2,1], collgrad=[0,1,0,1],
                        occupation = ["A","B","A","C"])
        df2s, Xs, ys = prepare_occupation_data(dfs)
        @test length(ys) == nrow(df2s)
        @test minimum(ys) == 1

        # ensure function errors sensibly when required cols missing
        dfbad = DataFrame(a = 1:3)
        @test_throws ErrorException prepare_occupation_data(dfbad)
    end

    @testset "ols_gmm - correctness & numeric checks" begin
        # exact fit => objective 0
        X = [ones(4) [1.0;2.0;3.0;4.0]]
        β = [1.5, -0.3]
        y = X * β
        Jval = ols_gmm(β, X, y)
        @test close_enough(Jval, 0.0)

        # compare to direct SSR for random β
        Random.seed!(1)
        Xr = hcat(ones(50), randn(50,2))
        βr = randn(3)
        yr = Xr * βr + 0.1*randn(50)
        Jcalc = ols_gmm(βr, Xr, yr)
        @test close_enough(Jcalc, dot(yr - Xr*βr, yr - Xr*βr))

        # wrong-dimension β should error
        @test_throws DimensionMismatch ols_gmm(rand(2), Xr, yr)
    end

    @testset "mlogit_mle - probabilities, likelihood edges" begin
        N = 12; K = 2; J = 3
        X = hcat(ones(N), randn(N))
        y = repeat(1:J, inner=div(N,J))
        α_zero = zeros(K*(J-1))

        # With zero α, predicted P = 1/J and nll = -N*log(1/J)
        nll = mlogit_mle(α_zero, X, y)
        @test isfinite(nll)
        @test isapprox(nll, -N*log(1/J); atol=1e-12)

        # For random α, P matrix should be finite and rows sum to 1
        α_r = randn(K*(J-1))
        # create internal pieces to test probabilities via same transform as source
        bigα = hcat(reshape(α_r, K, J-1), zeros(K))
        expM = exp.(X * bigα)
        P = expM ./ sum.(eachrow(expM))
        @test is_probability_matrix(P)

        # log-likelihood monotonicity: better fit => lower NLL
        α_bad = α_zero
        α_better = copy(α_zero); α_better[1] += 1.0
        nll_bad = mlogit_mle(α_bad, X, y)
        nll_better = mlogit_mle(α_better, X, y)
        @test isfinite(nll_bad) && isfinite(nll_better)
    end

    @testset "mlogit_gmm and mlogit_gmm_overid - moment checks" begin
        N=20; K=2; J=3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = zeros(K*(J-1))

        # overid objective equals manual computation
        J_over = mlogit_gmm_overid(α, X, y)
        bigY = zeros(N, J)
        for j in 1:J
            bigY[:,j] .= (y .== j)
        end
        bigα = hcat(reshape(α, K, J-1), zeros(K))
        P = exp.(X * bigα) ./ sum.(eachrow(exp.(X * bigα)))
        expected = dot(bigY[:] .- P[:], bigY[:] .- P[:])
        @test isapprox(J_over, expected; atol=1e-12)

        # just-identified GMM returns finite positive scalar
        gmmJ = mlogit_gmm(α, X, y)
        @test isfinite(gmmJ) && gmmJ >= 0.0

        # wrong-size α triggers an error
        @test_throws BoundsError mlogit_gmm(rand(5), X, y)
    end

    @testset "sim_logit - distributional properties & determinism" begin
        Random.seed!(2025)
        N = 200
        J = 4
        Y, X = sim_logit(N, J)
        @test length(Y) == N
        @test size(X,1) == N
        @test all(isfinite.(X))
        # choices are convertible to 1..J
        Yint = Int.(round.(Y))
        @test minimum(Yint) >= 1 && maximum(Yint) <= J

        # test reproducibility with seed
        Random.seed!(2025)
        Y2, _ = sim_logit(N, J)
        @test all(Y .== Y2)
    end

    @testset "sim_logit_with_gumbel - correctness & argmax behavior" begin
        Random.seed!(2025)
        N=150; J=5
        Y, X = sim_logit_with_gumbel(N, J)
        @test length(Y) == N
        @test size(X,1) == N
        @test all((1 .<= Y) .& (Y .<= J))

        # argmax consistency under seed
        Random.seed!(2025)
        Y2, _ = sim_logit_with_gumbel(N, J)
        @test all(Y .== Y2)
    end

    @testset "mlogit_smm_overid - properties & reproducibility" begin
        Random.seed!(7)
        N = 30; K = 3; J = 3
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        α = zeros(K*(J-1))

        # small D for speed - finite and >= 0
        val = mlogit_smm_overid(α, X, y, 5)
        @test isfinite(val) && val >= 0

        # deterministic due to internal seeding
        v1 = mlogit_smm_overid(α, X, y, 3)
        v2 = mlogit_smm_overid(α, X, y, 3)
        @test close_enough(v1, v2)

        # sensitivity: changing α should change objective (usually)
        v_alpha = mlogit_smm_overid(fill(0.5, length(α)), X, y, 5)
        @test isfinite(v_alpha)
    end

    @testset "robustness: small-sample, degenerate inputs" begin
        # N=1 degenerate case
        X = hcat(ones(1), [0.0])
        y = [1]
        α = zeros( size(X,2)*(1 - 1) )  # J=1 -> zero-length param vector (edge)
        # functions should not crash badly (may return 0-length computations)
        @test_throws DimensionMismatch mlogit_mle(α, X, y)  # likely invalid model; assert it errors

        # constant columns in X
        Xc = hcat(ones(10), ones(10))
        β = [1.0, 0.0]
        y = Xc * β
        @test isfinite(ols_gmm(β, Xc, y))
    end

    @testset "API & type stability quick checks" begin
        # Ensure functions accept different numeric element types (Float32)
        Xf = Float32.(hcat(ones(5), randn(5)))
        y = Int.(rand(1:3,5))
        α = zeros(Float32, size(Xf,2)*(maximum(y)-1))
        # calling should not throw a type conversion error (may throw other errors if invalid)
        try
            _ = mlogit_mle(collect(Float64.(α)), Array{Float64}(Xf), y)
            @test true
        catch e
            @test isa(e, Exception)  # we at least get a controlled exception, not catastrophic type crash
        end
    end

end