##########################################################
# BLP - Demand side with X as instruments
# Johanna Rayl
##########################################################

#cd("/Users/johannarayl/Dropbox/Second Year/IO 1/PS4")
cd("/home/jmr9694/IO1")

using MAT, DataFrames, LinearAlgebra, KNITRO, Random, Distributions, Plots, KernelDensity, JLD 

prod3 = matread("100markets3products.mat")
prod5 = matread("100markets5products.mat")
#prod310 = matread("10markets3products.mat")
#prod3 = matread("Simulation Data/100markets3products.mat")
#prod5 = matread("Simulation Data/100markets5products.mat")

M = 100 # for estimation
J = 3
R = 1000 # number of draws

z3 = prod3["Z"]
shares3 = prod3["shares"]
x3 = prod3["x1"]
w3 = prod3["w"]
p3 = prod3["P_opt"]
xi_all3 = prod3["xi_all"]
alphas3 = prod3["alphas"]
eta3 = prod3["eta"]

z5 = prod5["Z"]
shares5 = prod5["shares"]
x5 = prod5["x1"]
w5 = prod5["w"]
p5 = prod5["P_opt"]
xi_all5 = prod5["xi_all"]
alphas5 = prod5["alphas"]
eta5 = prod5["eta"]

## =====================
#        PART 1
## =====================
# Plot price distribution 
p_vec = reshape(p3, J*M, 1)
plot(range(extrema(p3)[1], extrema(p3)[2], length = 100),
    z -> pdf(kde(p3[1,:]), z), label = "Product 1")

plot!(range(extrema(p3)[1], extrema(p3)[2], length = 100),
    z -> pdf(kde(p3[2,:]), z), label = "Product 2")

plot!(range(extrema(p3)[1], extrema(p3)[2], length = 100),
    z -> pdf(kde(p3[3,:]), z), label = "Product 3", 
    title = "Prices - 3 Products")
savefig("prices.pdf")


# Calculate profits
mc = 2 .+ w3 .+ z3.+ eta3
pi = (p3 - reshape(mc, J, M)) .* shares3 # normalizing all markets to size 1

# Plot profits 
plot(range(extrema(pi)[1], extrema(pi)[2], length = 100),
    z -> pdf(kde(pi[1,:]), z), label = "Product 1")

plot!(range(extrema(pi)[1], extrema(pi)[2], length = 100),
    z -> pdf(kde(pi[2,:]), z), label = "Product 2")

plot!(range(extrema(pi)[1], extrema(pi)[2], length = 100),
    z -> pdf(kde(pi[3,:]), z), label = "Product 3",
    title = "Profits - 3 products")
savefig("profits.pdf")

# Calculate welfare
beta = [5,1,1]
alpha_p = 1
sigma = 1
p = 1000 # 1000 simulated people per market
Random.seed!(123)
eps = rand(GeneralizedExtremeValue(0,1,0), (J*M, p))
nu = exp.(randn(M,p))
nu = repeat(nu, inner = (J,1))
U = x3 * beta - alpha_p .* reshape(p3, J*M, 1) .- sigma * nu .+ xi_all3 .+ eps
U = reshape(U, J, M, p)

choice = zeros(M, p)
Utils = zeros(M, p)
for i in 1:M
    for j in 1:p
        max = findmax(U[:,i,j])
        choice[i,j] = max[2] 
        Utils[i,j] = max[1]
        if max[1] < 0
            Utils[i,j] = 0
        end
    end
end

Utils = reshape(Utils, M*p)

# Plot welfare 
plot(range(extrema(Utils)[1], extrema(Utils)[2], length = 100),
    z -> pdf(kde(Utils[:]), z), label = "",
    title = "Welfare - 3 Products")
savefig("welfare.pdf")

## =====================
#        PART 2.1
## =====================
E_xi_x = (1/(J*M)) * sum(xi_all3 .* x3, dims=1) # E[Xi * X]
E_xi_p = (1/(J*M)) * sum(xi_all3 .* reshape(p3, J*M, 1), dims=1) # E[Xi * p]
pbar = zeros(J,M)
for i = 1:J
    for j = 1:M
        p3_x = p3[:, 1:end .!= j]
        pbar[i,j] = (1/99) .* sum(p3_x[i,:])
    end
end
E_xi_pbar = (1/(J*M)) * sum(xi_all3 .* reshape(pbar, J*M,1), dims=1)


## =====================
#        PART 2.2
## =====================

# Random draws of nu 
Random.seed!(123)

ν = exp.(randn(M,R))
ν = repeat(ν, inner = (J,1))
ν = reshape(ν, J, M, R)

# Demand side instruments 
z1 = x3 # X as instrument for X
z2 = ones(J,M, size(x3,2)) # sum of other firms' characteristics
x_reshape = reshape(x3, J, M, :)

for i in 1:M
    for l in 1:3
        z2[1,i,l] = x_reshape[2,i,l] + x_reshape[3,i,l]
        z2[2,i,l] = x_reshape[1,i,l] + x_reshape[3,i,l]
        z2[3,i,l] = x_reshape[1,i,l] + x_reshape[2,i,l]
    end
end

z2 = reshape(z2, J*M, :)
z2 = z2[:,2:3] # first column is colinear with constant 

Z = hcat(z1, z2)

p_vec = reshape(p3, M*J, :)
X = hcat(p_vec, x3) # concatenate price and characteristics

P_z = Z * inv(Z' * Z) * Z' # instrument projection matrix

Q = size(Z,2) # number of moments 

function mkt_share(θ) # constraint 1: market shares

    δ = θ[1:J*M]
    σ = θ[end]

    delta_jm = reshape(δ, J, M)

    sum_exp = sum( exp.(delta_jm .+ σ .* p3 .* ν), dims=1) # (1 x M x R)
    share_R = exp.(delta_jm .+ σ .* p3 .* ν) ./ (1 .+ sum_exp) # (J x M x R)
    shares = (1/R) * sum(share_R, dims=3) # (J x M)

    diff1 = reshape(shares, J*M, :) - reshape(shares3, J*M,:)

    return diff1
    
end

function moments(θ) # constraint 2: moments 

    δ = θ[1:J*M]
    η = θ[J*M + 1: J*M + Q]

    # Concentrate out alpha, beta with TSLS
    xi = (I - X * inv(X' * P_z * X) * (X' * P_z)) * δ

    moms = xi .* Z

    return moms

end

function grad(θ) # compute gradient of GMM objective 

    η = θ[J*M + 1: J*M + Q]
    grad_η = 2 * W * η

    return grad_η       

end

function cons_grad(θ) # gradient of constraints 

    δ = θ[1:J*M]
    η  = θ[J*M + 1: J*M + Q]
    σ = θ[end]
    delta_jm = reshape(δ, J, M)

    # Compute share objects 
    exp_u = exp.(delta_jm .+ σ .* p3 .* ν)  # (J x M x R)
    share_R = exp.(delta_jm .+ σ .* p3 .* ν) ./ (1 .+ sum(exp_u, dims=1)) # (J x M x R)

    ds_dσ = (1/R) * sum(share_R .* (p3 .* ν .- sum(share_R .* p3 .* ν, dims=1)), dims=3) # (J x M)
    ds_dσ = reshape(ds_dσ, J*M) # (J*M x 1)

    ds_dδ = zeros(J, M, J, M) # shares along dims 1 & 2, parameters (deltas) along dims 3 & 4 
    for j in 1:J 
        for k in 1:J
            for m in 1:M
            ds_dδ[j,m,k,m] = -1 .* (1/R) .* sum(share_R[j,m,:] .* share_R[k,m,:])
                if j == k
                    ds_dδ[j,m,k,m] = (1/R) .* sum(share_R[j,m,:] .* (1 .- share_R[j,m,:]))
                end
            end
        end
    end

    ds_dδ = reshape(ds_dδ, J*M, J*M) 

    # Compute moment function objects
    dg_dδ = (1/(J*M)) .* transpose(Z) * (I - X * inv(X' * P_z * X) * X' * P_z) 
    dg_dη = zeros(Q,Q)
    for i = 1:Q 
        dg_dη[i,i] = -1
    end

    # Combine 
    grad_s = hcat(ds_dδ, zeros(J*M,Q), ds_dσ) 
    grad_g = hcat(dg_dδ, dg_dη, zeros(Q,1))

    return grad_s, grad_g

end

function callbackEvalFC(kc, cb, evalRequest, evalResult, userParams) # objective function and constraints
    θ = evalRequest.x
    
    η  = θ[J*M + 1: J*M + Q]
   
    obj = η' * W * η 

    evalResult.obj[1] = obj[1]

    c1 = mkt_share(θ)
    c2 = transpose((1/ (J*M)) * sum(moments(θ), dims=1)) .- η

    for i  in 1:J*M 
        evalResult.c[i] = c1[i]
    end

    for i in 1:Q
        evalResult.c[J*M+i] = c2[i]
    end

    return 0
end

function callbackEvalGA(kc, cb, evalRequest, evalResult, userParams)

    θ = evalRequest.x 

    grad_η = grad(θ)
    (grad_s, grad_g) = cons_grad(θ)

    # Objective gradient
    for i in 1:J*M 
        evalResult.objGrad[i] = 0 # wrt δ
    end

    for i in 1:Q
        evalResult.objGrad[J*M + i] = grad_η[i] # wrt η  
    end

    evalResult.objGrad[end] = 0 # wrt σ

    # Constraints 1-300 (shares) gradient 
    for j in 1:J*M
        for i in 1:J*M+Q+1
            evalResult.jac[(j-1)* (J*M+Q+1) + i] = grad_s[j, i] 
        end
    end

    # Constraints 301-305 (moments) gradient 
    for j in 1:Q 
        for i in 1:J*M+Q+1 
            evalResult.jac[J*M*(J*M+Q+1) + (j-1)*(J*M+Q+1) + i] = grad_g[j,i] 
        end
    end

    return 0
end

function estimate(start)

    # Initialize solver
    kc = KNITRO.KN_new() # new solver instance 

    KNITRO.KN_add_vars(kc, n) 

    # Set start values
    KNITRO.KN_set_var_primal_init_values(kc, start)

    # Set parameter bounds - none
    KNITRO.KN_set_var_lobnds(kc, -KNITRO.KN_INFINITY*ones(n))
    KNITRO.KN_set_var_upbnds(kc, KNITRO.KN_INFINITY*ones(n))

    KNITRO.KN_set_var_lobnds(kc, -KNITRO.KN_INFINITY*ones(n))
    KNITRO.KN_set_var_upbnds(kc, KNITRO.KN_INFINITY*ones(n))

    # Define constraints
    m = J*M + Q
    KNITRO.KN_add_cons(kc, m) # add 2 constraints
    KNITRO.KN_set_con_eqbnds(kc, ones(m)*0)

    cIndices = Array{Int32}(undef, m) # constraint indices
    for i in 1:m
        cIndices[i] = i-1
    end

    objGradIndexVarsCB = Array{Int32}(undef, n)
    for i = 1:n
        objGradIndexVarsCB[i] = i-1
    end

    # Constraint Jacobian non-zero structure for callback
    jacIndexConsCB = Array{Int32}(undef, n*m)
    for i = 1:m
        for j = 1:n 
            jacIndexConsCB[(i-1)*n + j] = i-1
        end
    end 
    jacIndexVarsCB = Array{Int32}(undef, n*m)
    for i = 1:m 
        for j = 1:n 
            jacIndexVarsCB[(i-1)*n + j] = j-1
        end
    end

    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE) # minimizaton problem
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEvalFC)# function to evaluate 

    KNITRO.KN_set_cb_grad(kc, cb, callbackEvalGA,
                    nV=length(objGradIndexVarsCB),
                    objGradIndexVars=objGradIndexVarsCB,
                    jacIndexCons=jacIndexConsCB,
                    jacIndexVars=jacIndexVarsCB)

    nStatus = KNITRO.KN_solve(kc) # solve the model 
    nStatus, objSol, θ, lambda_ = KNITRO.KN_get_solution(kc) # store results 
    KNITRO.KN_free(kc) # end solver instance 

    delta_hat = θ[1:J*M]
    coef = vcat(inv(X' * P_z * X) * (X' * P_z * delta_hat), θ[end])

    print(coef)

    return θ, coef
end

# Number of parameters in estimation 
n = J*M + Q + 1

# Start value array
start2 = vcat(3 * ones(J*M), 0 * ones(Q), -1)
start3 = 0 * ones(n)
start4 = 2*ones(n)
start5 = randn(n)
#start = hcat(start1, start2, start3, start4, start5)

# Initial weight matrix 
W = zeros(Q,Q)
for i  in 1:Q 
    W[i,i] = 1
end

# Start value tests
#for i in 1:5
    #(theta, coef) = estimate(start[:,i])
#end

# Initial parameter estimates
(theta_1, coef_1) = estimate(start3)

# Optimal weight matrix 
mom = moments(theta_1)
cov = zeros(Q, Q, J*M)
for i in 1:Q
    for j in 1:Q
        cov[i,j,:] = reshape((mom[:,i] .- sum(mom[:,i])) .* (mom[:,j] .- sum(mom[:,j])), (1,1,J*M))
    end
end
W = inv( dropdims( (1/(J*M)) * sum(cov, dims=3); dims=3) ) # (Q x Q)

# Second step parameter estimates 
(theta_2, coef_2) = estimate(theta_1)

# Compute standard errors for 5 parameters of interest 
dxi_dθ1 = -1 * hcat(p_vec, x3)
(grad_s, grad_g) = cons_grad(theta_2)
dxi_dθ2 = -1 * inv(grad_s[:,1:J*M]) * grad_s[:,end]
dxi_dθ = hcat(dxi_dθ1, dxi_dθ2) # (J*M x 5)
dg_dθ = ones(Q, 5, J*M) 
for i in 1:J*M
    dg_dθ[:,:,i] = Z[i,:] * transpose(dxi_dθ[i,:]) # outer product of z_jm and dxi_dθ_jm 
end
dg_dθ = dropdims((1/ (J*M)) * sum(dg_dθ, dims=3), dims=3) # (Q x 5) gradient of moment function 

mom = moments(theta_2)
cov = zeros(Q, Q, J*M)
for i in 1:Q
    for j in 1:Q
        cov[i,j,:] = reshape((mom[:,i] .- sum(mom[:,i])) .* (mom[:,j] .- sum(mom[:,j])), (1,1,J*M)) # moment covariance 
    end
end

B = dropdims((1/(J*M)) * sum(cov, dims=3); dims=3) # meat of sandwich 
V = (inv(dg_dθ' * W * dg_dθ)) * (dg_dθ' * W * B * W * dg_dθ) * (inv(dg_dθ' * W * dg_dθ))
se = (1/ sqrt(J*M)) * hcat(sqrt(V[1,1]), sqrt(V[2,2]), sqrt(V[3,3]), sqrt(V[4,4]), sqrt(V[5,5]))

print(se)

save("est_x.jld", "est_x", theta_2)

# ========================================
# Demand elasticities, profits, welfare 
# ========================================

function ds_dp(θ) # derivative of shares wrt prices 

    δ = θ[1:J*M]
    σ = θ[end]

    delta_jm = reshape(δ, J, M)

    sum_exp = sum( exp.(delta_jm .+ σ .* p3 .* ν), dims=1) # (1 x M x R)
    share_R = exp.(delta_jm .+ σ .* p3 .* ν) ./ (1 .+ sum_exp) # (J x M x R)
    shares = (1/R) * sum(share_R, dims=3) # (J x M)

    alph = inv(X' * P_z * X) * (X' * P_z * δ)
    alph = alph[1]

    ds_dp = ones(J,J,M)
    for i in 1:J 
        for k in 1:J 
            ds_dp[i,k,:] = (-1/R) * sum( share_R[i,:,:] .* share_R[k,:,:] .* (alph .+ σ .* ν[i,:,:]), dims=2) 
            if i == k 
                ds_dp[i,k,:] = (1/R) * sum( (alph .+ σ * ν[i,:,:]) .* share_R[i,:,:] .* (1 .-share_R[i,:,:] ) , dims=2)
            end
        end
    end

    return ds_dp, shares 

end 

# Elasticities 
theta_2 = load("est_x.jld")["est_x"]
coef_2 = vcat(inv(X' * P_z * X) * (X' * P_z * theta_2[1:J*M]), theta_2[end])

(dsdp, shares) = ds_dp(theta_2)
dsdp_avg = (1/M) * sum(dsdp, dims=3) # report average over all markets 
print(dsdp_avg)

# True elasticities 
delta_true = x3 * beta - p_vec + xi_all3
theta_true = vcat(delta_true, ones(Q), -1)
(dsdp_true, shares_true) = ds_dp(theta_true)
dsdp_avg_true = (1/M) * sum(dsdp_true, dims=3) # report average over all markets 
print(dsdp_avg_true)

# Profits 
Δ = zeros(J,J,M)
for i in 1:J 
    Δ[i,i,:] = -1 .* dsdp[i,i,:]
end

mc_est = ones(J,M) # Compute marginal costs 
for i in 1:M 
    mc_est[:,i] = p3[:,i] - inv(Δ[:,:,i]) * shares[:,i]
end

pi_est = (p3 - mc_est) .* shares 
tot_pi_est = sum(pi_est, dims=2)
tot_pi = sum(pi, dims=2)

# CS 
δ = theta_2[1:J*M]
xi = (I - X * inv(X' * P_z * X) * (X' * P_z)) * δ
U_est = x3 * coef_2[2:4] .+ coef_2[1] * p_vec .+ coef_2[end] * nu .+ xi .+ eps 

U_est = reshape(U_est, J, M, R)

choice = zeros(M, R)
Utils_est = zeros(M, R)
for i in 1:M
    for j in 1:R
        max = findmax(U_est[:,i,j])
        choice[i,j] = max[2] 
        Utils_est[i,j] = max[1]
        if max[1] < 0
            Utils_est[i,j] = 0
        end
    end
end

Utils_est = reshape(Utils_est, M*R)

# Plot welfare 
plot(range(extrema(Utils)[1], extrema(Utils)[2], length = 100),
    z -> pdf(kde(Utils[:]), z), label = "True")
plot!(range(extrema(Utils_est)[1], extrema(Utils_est)[2], length = 100),
    z -> pdf(kde(Utils_est[:]), z), label = "Estimated",
    title = "Welfare")
savefig("welfare_compare.pdf")