##################################################################
# BLP - Joint demand and supply
# Johanna Rayl
##################################################################

#cd("/Users/johannarayl/Dropbox/Second Year/IO 1/PS4")
cd("/home/jmr9694/IO1")

using MAT, DataFrames, LinearAlgebra, KNITRO, Random, Distributions, Plots, JLD

prod3 = matread("100markets3products.mat")
#prod3 = matread("Simulation Data/100markets3products.mat")

M = 100
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

z3 = w3 # add cost shifters 

Z = hcat(z1, z2, z3)

p_vec = reshape(p3, M*J, :)
X = hcat(p_vec, x3) # concatenate price and characteristics

P_z = Z * inv(Z' * Z) * Z' # instrument projection matrix

Q = 2 * size(Z,2) # number of moments (supply and demand)

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

## =====================
#        PART 3.2 b 
## =====================

# In oligopoly, only elements on diagonal of Δ are non-zero
est = load("est_wx.jld")["est_wx"]
(dsdp, shares) = ds_dp(est)
Δ = zeros(J,J,M)
for i in 1:J 
    Δ[i,i,:] = -1 .* dsdp[i,i,:]
end

# Compute marginal costs 
mc = ones(J,M)
for i in 1:M 
    mc[:,i] = p3[:,i] - inv(Δ[:,:,i]) * shares[:,i]
end

# Compare with true marginal costs 
mc_true = reshape(2 .+ w3 + z3 + eta3, J, M)

diff = ones(J,M)
for i = 1:J 
    for m = 1:M 
        diff[i,m] = abs(mc[i,m] - mc_true[i,m])
    end
end
avg_diff = (1/(J*M)) * sum(reshape(diff, J*M))

# Plot two distributions 
plot(range(extrema(mc)[1], extrema(mc)[2], length = 100),
    z -> pdf(kde(mc[:]), z), label = "Estimated")

plot!(range(extrema(mc_true)[1], extrema(mc_true)[2], length = 100),
    z -> pdf(kde(mc_true[:]), z), label = "True", 
    title = "Marginal Costs")
savefig("mc.pdf")

## ======================

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

function pc_moments(θ) # constraint 2: moments under perfect competition 

    δ = θ[1:J*M]
    η = θ[J*M + 1: J*M + Q]
    γ_0 = θ[J*M+Q+1]
    γ_1 = θ[J*M+Q+2]
    γ_2 = θ[J*M+Q+3]

    # Concentrate out alpha, beta with TSLS
    xi = (I - X * inv(X' * P_z * X) * (X' * P_z)) * δ

    mom_1 = xi .* Z

    # Supply moments
    mom_2 = (p_vec .- γ_0 .- γ_1 * w3 .- γ_2 * z3 ) .* Z 

    moms = hcat(mom_1, mom_2)

    return moms

end

function oli_moments(θ) # constraint 2: moments under oligopoly 

    δ = θ[1:J*M]
    η = θ[J*M + 1: J*M + Q]
    γ_0 = θ[J*M+Q+1]
    γ_1 = θ[J*M+Q+2]
    γ_2 = θ[J*M+Q+3]

    # Concentrate out alpha, beta with TSLS
    xi = (I - X * inv(X' * P_z * X) * (X' * P_z)) * δ

    mom_1 = xi .* Z

    # Supply moments
    (dsdp, shares) = ds_dp(θ)
    Δ = zeros(J,J,M)
    for i in 1:J 
        Δ[i,i,:] = -1 .* dsdp[i,i,:]
    end
    
    # Compute marginal costs 
    mc = ones(J,M)
    for i in 1:M 
        mc[:,i] = p3[:,i] - inv(Δ[:,:,i]) * shares[:,i]
    end

    mom_2 = (reshape(mc, J*M) .- γ_0 .- γ_1 * w3 .- γ_2 * z3) .* Z

    mom = hcat(mom_1, mom_2)

    return mom

end

function coll_moments(θ) # constraint 2: moments under perfect collusion

    γ_0 = θ[J*M+Q+1]
    γ_1 = θ[J*M+Q+2]
    γ_2 = θ[J*M+Q+3]

    # Concentrate out alpha, beta with TSLS
    xi = (I - X * inv(X' * P_z * X) * (X' * P_z)) * δ

    mom_1 = xi .* Z

    # Supply moments
    (dsdp, shares) = ds_dp(θ)
    Δ = -1 * dsdp 
    
    # Compute marginal costs 
    mc = ones(J,M)
    for i in 1:M 
        mc[:,i] = p3[:,i] - inv(Δ[:,:,i]) * shares[:,i]
    end

    mom_2 = (reshape(mc, J*M) .- γ_0 .- γ_1 * w3 .- γ_2 * z3) .* Z

    mom = hcat(mom_1, mom_2)

    return mom

end

function callbackEvalFC(kc, cb, evalRequest, evalResult, userParams) # objective function and constraints
    θ = evalRequest.x
    
    η  = θ[J*M + 1: J*M + Q]
   
    obj = η' * W * η 

    evalResult.obj[1] = obj[1]

    c1 = mkt_share(θ)
    c2 = transpose((1/ (J*M)) * sum(oli_moments(θ), dims=1)) .- η # enter supply conduct type here 

    for i  in 1:J*M 
        evalResult.c[i] = c1[i]
    end

    for i in 1:Q
        evalResult.c[J*M+i] = c2[i]
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

    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE) # minimizaton problem
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEvalFC)# function to evaluate 

    nStatus = KNITRO.KN_solve(kc) # solve the model 
    nStatus, objSol, θ, lambda_ = KNITRO.KN_get_solution(kc) # store results 
    KNITRO.KN_free(kc) # end solver instance 

    delta_hat = θ[1:J*M]
    coef = vcat(inv(X' * P_z * X) * (X' * P_z * delta_hat), θ[end])

    print(coef)

    return θ, coef
end

# Number of parameters in estimation 
n = J*M + Q + 4

# Start value array
start = zeros(n)

# Initial weight matrix 
W = zeros(Q,Q)
for i  in 1:Q 
    W[i,i] = 1
end

# Initial parameter estimates
(theta_1, coef_1) = estimate(start)

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
