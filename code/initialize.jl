using Pkg
# Pkg.add(["Parameters", "Distributions", "Optim", "DataFrames", "LinearAlgebra", "StatFiles"])
using Distributions
using Optim
using Parameters
using DataFrames
using LinearAlgebra
using StatFiles


const DATA = DataFrame(load("proc/dataset_exp_complete_Xs_SATONLY.dta"))
const NX = 26
const TYPE_INDICES = [findall(DATA.hhsize_type .== Float64(i)) for i in 1:3]
const MU = zeros(3)  # Mean vector for the multivariate normal distribution
const X_ALPHA_BETA_MAT = Matrix(DATA[:, ["one", ["x"*string(i) for i in 1:26]...]])
const X_ETA_MAT = Matrix(DATA[:, ["one", ["x"*string(i) for i in 1:27]...]])