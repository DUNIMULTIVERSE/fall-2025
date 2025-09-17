using  Optim, DataFrames, CSV, HTTP, GLM,LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
function allwrap()
cd(@_DIR_)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))

result_better = optimize(minusf, [-7.0], BFGS())
println(result_better)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end
"""
function ols2(beta, X, y)
    ssr = sum((y.-X*beta).^2)
    return ssr
end
"""
beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
println("Ols closed form:",bols)
# Standard Errors
N = size(X,1)
K = size(X,2)
MSE = sum((y.-X*bols).^2)/(N-K)
var_bols = MSE*inv(X'*X)
se_bols = sqrt.(diag(var_bols))
println("Standard Errors:",se_bols)

df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function logit(alpha, X, d)
    pi1 = exp.(X*alpha)./(1 .+ exp.(X*alpha))
    ll = -sum((d.==1).* log.(pi1) .+ (d.==0).* log.(1 .-pi1))
    return ll
end
alpha_hat_logit = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println(alpha_hat_glm)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# 5 (Dependent variable occupation)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) 

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

K = size(X,2)
J = length(unique(y))

function mlogit(alpha, X, d)


N = length(y)
bigY = zeros(N,J)
for j=1:J
bigY[:,j] .= (y .== j)
end
bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
num = zeros(N,J)
den = zeros(N)
for j=1:J
num[:,j] = exp.(X*bigAlpha[:,j])
den .+= num[:,j]
end

P = num ./ repeat(den, 1, J)
loglike = -sum(bigY .* log.(P))

return loglike
end

alpha_start = randn(K * (J - 1)) 
alpha_hat_optim = optimize(α -> mlogit(α, X, y),alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 1000))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)
end
allwrap()

