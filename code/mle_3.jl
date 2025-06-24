
using Pkg
# Pkg.add(["Parameters", "Distributions", "Optim", "DataFrames", "LinearAlgebra", "StatFiles"])
using Distributions
using Optim
using Parameters
using DataFrames
using LinearAlgebra
using StatFiles

include("basic_functions.jl")

DATA = DataFrame(load("/home/s/spandita/calvi/proc/dataset_exp_complete_Xs_SATONLY.dta"))
NX = 26


αm1::Vector{Float64}, αm2::Vector{Float64}, αm3::Vector{Float64}, 
αf1::Vector{Float64}, αf2::Vector{Float64}, αf3::Vector{Float64},
αc1::Vector{Float64}, αc2::Vector{Float64}, αc3::Vector{Float64} = ntuple(_ -> 0.01*ones(NX+1),9);

βm::Vector{Float64}, βf::Vector{Float64}, βc::Vector{Float64} = ntuple(_ -> 0.01*ones(NX+1),3);

ηm1::Vector{Float64}, ηm2::Vector{Float64}, ηm3::Vector{Float64}, 
ηf1::Vector{Float64}, ηf2::Vector{Float64}, ηf3::Vector{Float64} = ntuple(_ -> 0.01*ones(NX+2),6);

Sigma = Matrix{Float64}(I(9))  # Initial guess for Sigma, a 9x9 identity matrix

function log_likelihood(errors::Matrix{Float64}, Sigma::Matrix{Float64})
    # errors is a matrix of size Nobs x 3, where each row is an error vector for a household
    # Sigma is a 9x9 matrix, which is a block diagonal matrix with 3x3 matrices for each household size type
    # The log likelihood is calculated as the sum of the log likelihoods of each error vector
    Nobs = size(errors, 1)
    Mu = zeros(3)  # Mean vector for the multivariate normal distribution
    # LL_array = zeros(Nobs, 3)  # Log likelihood array for the errors 
    ll = 0.0 
    for i in 1:Nobs
        type = Int(DATA.hhsize_type[i])
        # Get the 3x3 Sigma matrix for this type
        @views S = Sigma[(type-1)*3+1:type*3, (type-1)*3+1:type*3]
        ll += log(pdf(MvNormal(Mu, S), errors[i,:]))
    end
    return -ll  # Return the negative log likelihood
end

function optim_func(guess::Vector{Float64}, Sigma::Matrix{Float64})
    αm1 = guess[1:NX+1]
    αm2 = guess[NX+2:2*NX+2]
    αm3 = guess[2*NX+3:3*NX+3]
    αf1 = guess[3*NX+4:4*NX+4]
    αf2 = guess[4*NX+5:5*NX+5]
    αf3 = guess[5*NX+6:6*NX+6]
    αc1 = guess[6*NX+7:7*NX+7]
    αc2 = guess[7*NX+8:8*NX+8]
    αc3 = guess[8*NX+9:9*NX+9]
    # βm, βf, βc are the next 3*Nx + 3 parameters
    βm = guess[9*NX+10:10*NX+10]
    βf = guess[10*NX+11:11*NX+11]
    βc = guess[11*NX+12:12*NX+12]
    # ηm1, ηm2, ηm3, ηf1, ηf2, ηf3 are the next 6*(NX + 2) parameters
    ηm1 = guess[12*NX+13:13*NX+14]
    ηm2 = guess[13*NX+15:14*NX+16]
    ηm3 = guess[14*NX+17:15*NX+18]
    ηf1 = guess[15*NX+19:16*NX+20]
    ηf2 = guess[16*NX+21:17*NX+22]
    ηf3 = guess[17*NX+23:18*NX+24]
    errs = gen_errors(αm1, αm2, αm3, αf1, αf2, αf3, 
                      αc1, αc2, αc3, βm, βf, βc, 
                      ηm1, ηm2, ηm3, ηf1, ηf2, ηf3,
                      DATA, NX)
    if isnothing(errs)
        return 1e100  # Return a large value if errors are not generated
    end
    
    return log_likelihood(errs, Sigma)
end


sigma_diff = 1e6
while sigma_diff > 1e-6
    optim_f = guess -> optim_func(guess, Sigma)
    res = optimize(optim_f, vcat(αm1, αm2, αm3, αf1, αf2, αf3, 
                                αc1, αc2, αc3, βm, βf, βc, 
                                ηm1, ηm2, ηm3, ηf1, ηf2, ηf3), 
                                NelderMead(), Optim.Options(iterations=50))
    guess = res.minimizer
    αm1 = guess[1:NX+1]
    αm2 = guess[NX+2:2*NX+2]
    αm3 = guess[2*NX+3:3*NX+3]
    αf1 = guess[3*NX+4:4*NX+4]
    αf2 = guess[4*NX+5:5*NX+5]
    αf3 = guess[5*NX+6:6*NX+6]
    αc1 = guess[6*NX+7:7*NX+7]
    αc2 = guess[7*NX+8:8*NX+8]
    αc3 = guess[8*NX+9:9*NX+9]
    βm = guess[9*NX+10:10*NX+10]
    βf = guess[10*NX+11:11*NX+11]
    βc = guess[11*NX+12:12*NX+12]
    ηm1 = guess[12*NX+13:13*NX+14]
    ηm2 = guess[13*NX+15:14*NX+16]
    ηm3 = guess[14*NX+17:15*NX+18]
    ηf1 = guess[15*NX+19:16*NX+20]
    ηf2 = guess[16*NX+21:17*NX+22]
    ηf3 = guess[17*NX+23:18*NX+24]
    errs = gen_errors(αm1, αm2, αm3, αf1, αf2, αf3, 
                    αc1, αc2, αc3, βm, βf, βc, 
                    ηm1, ηm2, ηm3, ηf1, ηf2, ηf3,
                    DATA, NX)
    if isnothing(errs)
        println("Errors not generated, skipping iteration")
        continue
    end
    errs_1 = errs[DATA.hhsize_type .== 1.0, :]
    errs_2 = errs[DATA.hhsize_type .== 2.0, :]
    errs_3 = errs[DATA.hhsize_type .== 3.0, :]
    sigma_1 = cov(errs_1)
    sigma_2 = cov(errs_2)
    sigma_3 = cov(errs_3)
    Sigma_new = [sigma_1 zeros(3,3) zeros(3,3);
                zeros(3,3) sigma_2 zeros(3,3);
                zeros(3,3) zeros(3,3) sigma_3]
    sigma_diff = maximum(abs.(Sigma_new .- Sigma))
    Sigma = Sigma_new
end

# Save the results to a file
using Serialization
serialize("mle_results.jls", (αm1, αm2, αm3, αf1, αf2, αf3, αc1, αc2, αc3, βm, βf, βc, ηm1, ηm2, ηm3, ηf1, ηf2, ηf3, Sigma))