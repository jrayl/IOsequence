##########################################################
# BLP - Demand side, Price instruments 
# Johanna Rayl
##########################################################

#cd("/Users/johannarayl/Dropbox/Second Year/IO 1/PS4")
cd("/home/jmr9694/IO1")

using MAT, DataFrames, LinearAlgebra, KNITRO, Random, Distributions, Plots

prod3 = matread("100markets3products.mat")

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

## =====================
#        PART 2.2.3
## =====================

# Random draws of nu 
Random.seed!(123)

ν = exp.(randn(M,R))
ν = repeat(ν, inner = (J,1))
ν = reshape(ν, J, M, R)

# Demand side instruments

pbar = zeros(J,M)
for i = 1:J
    for j = 1:M
        p3_x = p3[:, 1:end .!= j]
        pbar[i,j] = (1/99) .* sum(p3_x[i,:])
    end
end

Z = hcat(x3, reshape(p3, J*M, 1), reshape(pbar, J*M, 1)) # Hausman instruments 

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
start = hcat(start1, start2, start3, start4, start5)

# Initial weight matrix 
W = zeros(Q,Q)
for i  in 1:Q 
    W[i,i] = 1
end

# Initial parameter estimates
(theta_1, coef_1) = estimate(start1)

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
