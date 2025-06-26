include("initialize.jl")
include("basic_functions.jl")

αm1, αm2, αm3, 
αf1, αf2, αf3,
αc1, αc2, αc3 = ntuple(_ -> 0.01*ones(NX+1),9);

βm, βf, βc = ntuple(_ -> 0.01*ones(NX+1),3);

ηm1, ηm2, ηm3, 
ηf1, ηf2, ηf3 = ntuple(_ -> 0.01*ones(NX+2),6);

Sigma = Matrix{Float64}(I(9))  # Initial guess for Sigma, a 9x9 identity matrix

function log_likelihood(errors::Matrix{Float64}, Sigma::Matrix{Float64})
    # errors is a matrix of size Nobs x 3, where each row is an error vector for a household
    # Sigma is a 9x9 matrix, which is a block diagonal matrix with 3x3 matrices for each household size type
    # The log likelihood is calculated as the sum of the log likelihoods of each error vector
    Nobs = size(errors, 1)
    # Mu = zeros(3)  # Mean vector for the multivariate normal distribution
    # LL_array = zeros(Nobs, 3)  # Log likelihood array for the errors 
    ll = 0.0 
     # Process each type separately to avoid repeated indexing
    for type in 1:3
        S = @view Sigma[(type-1)*3+1:type*3, (type-1)*3+1:type*3]
        dist = MvNormal(MU, S)
        for idx in TYPE_INDICES[type]
            ll += logpdf(dist, @view errors[idx, :])
        end
    end
    return -ll  # Return the negative log likelihood
end

function optim_func(guess::Vector{Float64}, Sigma::Matrix{Float64})
    local αm1 = @view guess[1:NX+1]
    local αm2 = @view guess[NX+2:2*NX+2]
    local αm3 = @view guess[2*NX+3:3*NX+3]
    local αf1 = @view guess[3*NX+4:4*NX+4]
    local αf2 = @view guess[4*NX+5:5*NX+5]
    local αf3 = @view guess[5*NX+6:6*NX+6]
    local αc1 = @view guess[6*NX+7:7*NX+7]
    local αc2 = @view guess[7*NX+8:8*NX+8]
    local αc3 = @view guess[8*NX+9:9*NX+9]
    # βm, βf, βc are the next 3*Nx + 3 parameters
    local βm = @view guess[9*NX+10:10*NX+10]
    local βf = @view guess[10*NX+11:11*NX+11]
    local βc = @view guess[11*NX+12:12*NX+12]
    # ηm1, ηm2, ηm3, ηf1, ηf2, ηf3 are the next 6*(NX + 2) parameters
    local ηm1 = @view guess[12*NX+13:13*NX+14]
    local ηm2 = @view guess[13*NX+15:14*NX+16]
    local ηm3 = @view guess[14*NX+17:15*NX+18]
    local ηf1 = @view guess[15*NX+19:16*NX+20]
    local ηf2 = @view guess[16*NX+21:17*NX+22]
    local ηf3 = @view guess[17*NX+23:18*NX+24]
    errs = gen_errors(αm1, αm2, αm3, αf1, αf2, αf3, 
                      αc1, αc2, αc3, βm, βf, βc, 
                      ηm1, ηm2, ηm3, ηf1, ηf2, ηf3)
    if isnothing(errs)
        return 1e100  # Return a large value if errors are not generated
    end
    
    return log_likelihood(errs, Sigma)
end


sigma_diff = 1e6
param_diff = 1e6
Nmax = 1
n = 0
while sigma_diff > 1e-6 && n < Nmax && param_diff > 1e-6
    global n += 1
    optim_f = guess -> optim_func(guess, Sigma)
    res = optimize(optim_f, vcat(αm1, αm2, αm3, αf1, αf2, αf3, 
                   αc1, αc2, αc3, βm, βf, βc, 
                   ηm1, ηm2, ηm3, ηf1, ηf2, ηf3), LBFGS(), Optim.Options(iterations=100, show_trace=true, show_every=10))
    guess = res.minimizer
    global param_diff = maximum(abs.(guess .- vcat(αm1, αm2, αm3, αf1, αf2, αf3, 
                                            αc1, αc2, αc3, βm, βf, βc, 
                                            ηm1, ηm2, ηm3, ηf1, ηf2, ηf3)))
    global αm1 = @view guess[1:NX+1]
    global αm2 = @view guess[NX+2:2*NX+2]
    global αm3 = @view guess[2*NX+3:3*NX+3]
    global αf1 = @view guess[3*NX+4:4*NX+4]
    global αf2 = @view guess[4*NX+5:5*NX+5]
    global αf3 = @view guess[5*NX+6:6*NX+6]
    global αc1 = @view guess[6*NX+7:7*NX+7]
    global αc2 = @view guess[7*NX+8:8*NX+8]
    global αc3 = @view guess[8*NX+9:9*NX+9]
    global βm = @view guess[9*NX+10:10*NX+10]
    global βf = @view guess[10*NX+11:11*NX+11]
    global βc = @view guess[11*NX+12:12*NX+12]
    global ηm1 = @view guess[12*NX+13:13*NX+14]
    global ηm2 = @view guess[13*NX+15:14*NX+16]
    global ηm3 = @view guess[14*NX+17:15*NX+18]
    global ηf1 = @view guess[15*NX+19:16*NX+20]
    global ηf2 = @view guess[16*NX+21:17*NX+22]
    global ηf3 = @view guess[17*NX+23:18*NX+24]
    errs = gen_errors(αm1, αm2, αm3, αf1, αf2, αf3, 
                    αc1, αc2, αc3, βm, βf, βc, 
                    ηm1, ηm2, ηm3, ηf1, ηf2, ηf3)
    if isnothing(errs)
        println("Errors not generated, skipping iteration")
        continue
    end
    # Use pre-allocated arrays and in-place operations
    sigma_1 = cov(@view errs[TYPE_INDICES[1], :])
    sigma_2 = cov(@view errs[TYPE_INDICES[2], :])  
    sigma_3 = cov(@view errs[TYPE_INDICES[3], :])

    Sigma_new = [sigma_1 zeros(3,3) zeros(3,3);
                zeros(3,3) sigma_2 zeros(3,3);
                zeros(3,3) zeros(3,3) sigma_3]
    global sigma_diff = maximum(abs.(Sigma_new .- Sigma))
    @show n, sigma_diff
    global Sigma = Sigma_new
end


# # Save the results to a file
# using Serialization
# serialize("mle_results.jls", (αm1, αm2, αm3, αf1, αf2, αf3, αc1, αc2, αc3, βm, βf, βc, ηm1, ηm2, ηm3, ηf1, ηf2, ηf3, Sigma))

# println("Optimization complete. Final parameters: $αm1, $αm2, $αm3, $αf1, $αf2, $αf3, 
#         $αc1, $αc2, $αc3, $βm, $βf, $βc, 
#         $ηm1, $ηm2, $ηm3, $ηf1, $ηf2, $ηf3")
# println("Final Sigma: $Sigma")
