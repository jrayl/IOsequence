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
start1 = [-0.7612552918886281, 6.420419742682914, 5.063139795604555, 6.024477203489006, 5.170491749013278, 6.769320014566261, 4.237124087170005, 3.8183859746736992, 3.1750435994723074, 0.8803061986880992, 4.231229845722871, 2.995706774601763, 2.3964216581214255, 4.1165624113322075, 3.7831314115833057, 2.2261312127833084, 2.5266621912978295, 1.916525611389282, 2.840263231173593, 2.39333820843512, 1.8027778502764888, -1.886267110525436, 3.1406602806277575, 1.3062462513025732, 1.080295267189441, 5.289429271766431, 6.070943939397813, 0.8031984402184793, 5.1261238005856855, 0.8207886137892705, 1.5917906041793684, 2.5000537234153635, 0.955813574036357, 2.8902134753279003, 5.363494545044875, 4.633733587968754, 0.1972798110779276, 4.775384224490105, 3.605888125778149, 4.583460674755363, 3.7464562696909867, 4.105820767677408, 1.3289971944222756, 3.518635962529481, 4.54974122907651, 2.8159642400641296, 2.708480348147829, 4.7552721932814315, 1.6142186697509773, 6.548318723044667, 4.32849144353088, 2.409756615397192, 3.643507201570914, 1.2908438638646416, 4.846265788364125, 2.873274957842383, 4.758695708972545, 1.813437104054613, 2.715181091372587, 0.7882757017770416, 0.8566554147351031, 1.5201740131154269, 1.8504346814108792, 2.16139306847492, 4.524414311893914, 4.253171145933034, -2.068382102263921, 5.933972875709099, 5.4547421885518625, 3.8164808417762446, 3.5925124916059685, 3.4654330471567603, 3.0051562065702244, 5.095106161755267, 4.438116881135677, 6.387203047467579, 2.409194063966677, 7.778545254077945, 3.3709972801091452, 5.893596917463913, 5.1987307474962465, 2.2820280920025686, 5.63756166735119, 4.535933247758283, 0.742041400562297, 0.7158715372031701, 1.6812029066522958, 3.0056548683427953, 2.3504909905737796, 0.6692765665605062, 1.7314137013781703, 7.087402942669314, 5.518640175591901, 0.4677556194971042, 6.345590192821137, 5.896878449309694, 3.4424854380429983, -0.40645740545862574, 3.7183593265168775, 6.51211753930319, 6.662123226530775, 6.242363023657467, 4.145943935945431, 3.1785788814403504, 4.316133780475638, 3.3056109831542764, 0.6522775777557143, 3.792097755475666, -0.04126749533440243, -0.03255256089987127, 4.581652139364926, 0.7497254685172172, 2.7885058993169043, -0.10900554911200883, 5.100584067018079, 3.790149597758554, 4.575697781672172, 3.6461687857080545, 3.9504484783658653, 2.1524418088818176, 6.249331175190388, 2.0031653802336673, 6.057393256281988, 1.5649869408824508, 3.0820671923033167, 3.422825160677375, 2.9083381855960324, 2.92388132362133, 4.941761834656601, 2.524373831059459, 1.7098709081840373, 2.190945230701923, 2.597273636054122, -0.6768738771231424, 2.1639896200213253, 1.8989780482739267, 4.577766641679798, 3.677926054206612, 4.808632973075454, 3.150634334790689, 4.738157044130056, 2.402534853599962, 3.418651275260907, 1.3545205779663994, 6.492536281200736, 1.6254047609024913, 4.6117698512073995, 1.6506260216414392, 3.5723036464406706, 4.232105947216705, 3.2741740675800086, 4.6129265326736775, 1.9434274519902175, 2.7791311328543133, 4.2011094904742246, 5.14381544248126, 2.576171465507504, 3.9993197656956547, 4.383664653169247, 2.3718001342977875, 2.3479724526837416, 4.909766866519317, 6.501922978171673, 6.053661048722441, 1.6417226718482159, 4.990989277044323, 2.2597125243889753, 2.9262773679034684, 2.3075478607888837, 2.7123757612874706, 3.1413020889309284, 0.13107310428745136, 3.3696597521971965, 2.2135239807642337, 1.587196869556184, 3.6330591951383826, 3.695976975411296, 5.869717594657803, 4.786703191954981, 4.8431449342672295, 4.737251689668991, 6.1625604997962515, 3.049829158598945, 3.5983137966260967, 5.573327327379193, 5.9559433388983525, -0.35651464645135694, 4.775015992286047, 2.3118068620721903, 2.7777143476256865, 5.029502161740968, 1.544760743529958, 3.7321364989806973, 4.247886694470604, 0.32404654380203113, 2.0954425259260523, 5.320034480164163, 2.687247373955703, 3.428454883156937, 4.321575612682887, 5.7025086872732995, 4.114702738963324, 3.0278932579249496, 2.969491067479172, -0.09172944920682151, 0.9459222417322762, -0.40911751568686816, -0.5935422379867501, 3.284772375460481, -0.3332424125032206, 1.065279889346533, 2.7270467183516316, -0.340709924217488, 1.3336803502904306, 5.32058385650215, 2.354360210207724, 4.544664625524499, 4.39388978609421, 4.68987419566672, 5.069929305990333, 4.4020925552119765, 4.652225547489177, 2.0912272571625237, 1.9590801856371103, 2.1591697852907004, 3.331540657031259, 3.925690652834553, 0.07239087884027337, 2.0025066355796683, 2.8322142540556596, 3.6643009538817712, 2.1988030471102733, 0.36373469356800076, 0.33955390744682623, 2.2187069668375354, 2.4852128951539694, 4.586085210882502, 3.189803966791037, 5.539779405752418, 5.958563076089812, 3.413092865921528, 4.117346800441569, 1.9086502433843195, 3.828905622398901, 4.782568869948608, 2.9816195988908967, 1.688434407879407, 8.259121691456837, 5.746440233960659, 2.5247292024145596, -1.2958354302224702, -1.6010264977238646, 3.600566068740333, 3.926045412754655, 4.15643658878626, 4.043621918809597, 3.6884365393437326, 1.6451538564926138, 2.652634432517035, 3.6661235318412344, 3.017507774894308, 1.8021248746714336, 4.08147300746973, 0.7246937262212751, 2.1845375934039013, 0.9957227971208388, 2.2877605650548674, 6.161065654460079, 6.044947686366433, 5.468914741928911, 0.3629430972205258, 4.0076630585977835, 4.770236544471246, 2.0268711200688703, 3.4749860109011133, 2.285834538291315, -0.6282952458005964, 0.697096909187418, 0.874126029750997, 1.6462523075978956, 3.668495386404908, 2.552939033565481, -1.0624335332416863, 1.7098952681933355, 2.240834829867866, 0.5052359578521125, 3.0800701659916, 5.260126825592392, 3.46087631784126, 3.5529789257441737, 2.0178858652074454, 1.524761740234499, 3.479338437933625, 2.9059312960719166, 1.5668688275598937, 3.584420364759867, 2.703964948404282, 4.132528503956064, 5.709765674990795, 5.964993491682082, -1.6850718120397023e-13, -9.295047744190797e-14, -9.936201056340409e-15, 3.408316368443273e-8, -4.5251306139698376e-8, -1.157364114552657]
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
se = hcat(sqrt(V[1,1]), sqrt(V[2,2]), sqrt(V[3,3]), sqrt(V[4,4]), sqrt(V[5,5]))

print(se)
