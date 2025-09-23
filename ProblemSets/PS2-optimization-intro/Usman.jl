using Optim, DataFrames, CSV, HTTP, GLM, FreqTables

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(alpha, X, d)
    η = X * alpha
    p = 1 ./(1 .+ exp.(-η))
    loglike = -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    return loglike
end
alpha_hat_logit = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
println(alpha_hat_logit.minimizer)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
df.white = df.race .== 1  # Ensure the 'white' dummy is consistent
logit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("Logit coefficients via GLM: ", coef(logit_glm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, d)
    K = size(X, 2)
    J = 7  # number of categories after aggregation
    beta = reshape(alpha, K, J - 1)
    n = size(X, 1)

    utilities = X * beta
    utilities_full = hcat(utilities, zeros(n))

    denom = sum(exp.(utilities_full), dims=2)
    p_mat = exp.(utilities_full) ./ denom

    ind = zeros(n, J)
    for i in 1:n
        ind[i, d[i]] = 1.0
    end

    loglike = -sum(ind .* log.(p_mat))
    return loglike
end
init_params = zeros(size(X, 2) * 6)  # K x (J-1) parameters
result = optimize(a -> mlogit(a, X, y), init_params, LBFGS(),
                  Optim.Options(g_tol=1e-5, iterations=100_000))
println("Estimated coefficients:")
println(reshape(result.minimizer, size(X, 2), 6))  # reshape for better interpretability

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function run_all_questions()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    startval = rand(1)   # random starting value
    result = optimize(minusf, startval, BFGS())
    println("argmin (minimizer) is ", Optim.minimizer(result)[1])
    println("min is ", Optim.minimum(result))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y
    df.white = df.race .== 1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function logit(alpha, X, d)
        η = X * alpha
        p = 1 ./(1 .+ exp.(-η))
        loglike = -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
        return loglike
    end
    alpha_hat_logit = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
    println(alpha_hat_logit.minimizer)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    df.white = df.race .== 1   # Ensure the 'white' dummy is consistent
    logit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println("Logit coefficients via GLM: ", coef(logit_glm))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    freqtable(df, :occupation)  # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7
    freqtable(df, :occupation)  # problem solved

    X = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation

    function mlogit(alpha, X, d)
        K = size(X, 2)
        J = 7   # number of categories after aggregation
        beta = reshape(alpha, K, J - 1)
        n = size(X, 1)

        utilities = X * beta
        utilities_full = hcat(utilities, zeros(n))

        denom = sum(exp.(utilities_full), dims=2)
        p_mat = exp.(utilities_full) ./ denom

        ind = zeros(n, J)
        for i in 1:n
            ind[i, d[i]] = 1.0
        end

        loglike = -sum(ind .* log.(p_mat))
        return loglike
    end

    init_params = zeros(size(X, 2) * 6)   # K x (J-1) parameters
    result = optimize(a -> mlogit(a, X, y), init_params, LBFGS(),
                      Optim.Options(g_tol=1e-5, iterations=100_000))
    println("Estimated coefficients:")
    println(reshape(result.minimizer, size(X, 2), 6))
end