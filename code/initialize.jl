
using Pkg
# Pkg.add(["Parameters", "Distributions", "Optim", "DataFrames", "LinearAlgebra", "StatFiles"])
using Distributions
using Optim
using Parameters
using DataFrames
using LinearAlgebra
using StatFiles


@with_kw struct Primitives
    data::DataFrame = DataFrame(load("/home/s/spandita/calvi/proc/dataset_exp_complete_Xs_SATONLY.dta"))
    Nx = 26
end    


@with_kw mutable struct MLEParams
    αm1::Vector{Float64}
    αm2::Vector{Float64}
    αm3::Vector{Float64} 
    αf1::Vector{Float64}
    αf2::Vector{Float64}
    αf3::Vector{Float64}
    αc1::Vector{Float64}
    αc2::Vector{Float64}
    αc3::Vector{Float64}
    βm::Vector{Float64}
    βf::Vector{Float64}
    βc::Vector{Float64}  
    ηm1::Vector{Float64}  
    ηm2::Vector{Float64}  
    ηm3::Vector{Float64}  
    ηf1::Vector{Float64}  
    ηf2::Vector{Float64}  
    ηf3::Vector{Float64}  
    # S1::Matrix{Float64}  # 3x3 block diagonal matrix 
    # S2::Matrix{Float64}  # 3x3 block diagonal matrix
    # S3::Matrix{Float64}  # 3x3 block diagonal matrix
end

function Initialize(guess::Vector{Float64})
    prims = Primitives()
    Nx = prims.Nx
    # alpha and beta have Nx+1 parameters, and three types.
    αm1 = guess[1:Nx+1]
    αm2 = guess[Nx+2:2*Nx+2]
    αm3 = guess[2*Nx+3:3*Nx+3]
    αf1 = guess[3*Nx+4:4*Nx+4]
    αf2 = guess[4*Nx+5:5*Nx+5]
    αf3 = guess[5*Nx+6:6*Nx+6]
    αc1 = guess[6*Nx+7:7*Nx+7]
    αc2 = guess[7*Nx+8:8*Nx+8]
    αc3 = guess[8*Nx+9:9*Nx+9]
    # βm, βf, βc are the next 3*Nx + 3 parameters
    βm = guess[9*Nx+10:10*Nx+10]
    βf = guess[10*Nx+11:11*Nx+11]
    βc = guess[11*Nx+12:12*Nx+12]
    # ηm1, ηm2, ηm3, ηf1, ηf2, ηf3 are the next 6*(Nx + 2) parameters
    ηm1 = guess[12*Nx+13:13*Nx+14]
    ηm2 = guess[13*Nx+15:14*Nx+16]
    ηm3 = guess[14*Nx+17:15*Nx+18]
    ηf1 = guess[15*Nx+19:16*Nx+20]
    ηf2 = guess[16*Nx+21:17*Nx+22]
    ηf3 = guess[17*Nx+23:18*Nx+24]

    # Sigma is a 9x9 block diagonal matrix, the last 3 parameters are the diagonal blocks
    # s1, s2, s3 are all length 9
    # s1 = guess[18*Nx+25:18*Nx+33]  
    # s2 = guess[18*Nx+34:18*Nx+42]
    # s3 = guess[18*Nx+43:18*Nx+51]  
    # Sigma is a 9x9 block diagonal matrix such that s1, s2, s3 are its diagonal blocks
    # S1 = reshape(s1,3,3)
    # S2 = reshape(s2,3,3)
    # S3 = reshape(s3,3,3)
    # Sigma = [S1 zeros(3,3) zeros(3,3); 
    #          zeros(3,3) S2 zeros(3,3); 
    #          zeros(3,3) zeros(3,3) S3]
    mle_params = MLEParams(αm1, αm2, αm3, αf1, αf2, αf3, αc1, αc2, αc3, βm, βf, βc, ηm1, ηm2, ηm3, ηf1, ηf2, ηf3)
    return mle_params
end

@with_kw mutable struct errors 
    err_hat::Matrix{Vector{Float64}}
end