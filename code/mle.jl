using Distributions
using Optim

Nobs = 900  # Number of observations
# Parameters 
# 3 types of households, 3 members in each household
a = [0.5 0.3 0.2; 
     0.4 0.2 0.1; 
     0.3 0.1 0.5]  
b = [0.1, 0.8, 0.6]

# var-covar matrix for the error term 
# Sigma is positive definite
Sigma = [0.04 0.02 0.01; 
         0.02 0.03 0.015; 
         0.01 0.015 0.02]

Mu = zeros(3) # Mean of the error term
# Generate the error term from a multivariate normal distribution
error_term = rand(MvNormal(Mu, Sigma), Nobs)
# Log Expenditures come from N(1.0, 1.1^2) 
log_expenditures = rand(Normal(1.0, 1.1), Nobs)
w = zeros(Nobs, 3)

# i = 1:300 use first row of a, i = 301:600 use second row of a, i = 601:900 use third row of a
for i in 1:Nobs
    if i <= 300
        w[i, :] = a[1, :] .+ log_expenditures[i].*b .+ error_term[:, i]
    elseif i > 300 & i <= 600
        w[i, :] = a[2, :] .+ log_expenditures[i].*b .+ error_term[:, i]
    else
        w[i, :] = a[3, :] .+ log_expenditures[i].*b .+ error_term[:, i]
    end
end

# Data is a dataframe with four columns: budget shares = w1,w2,w3, exp = log_expenditures

using DataFrames
types = [ones(300), 2*ones(300), 3*ones(300)]
data = DataFrame(b1 = w[:,1], b2 = w[:,2], b3 = w[:,3], exp = log_expenditures)
data[!,"type"] = vcat(types...)
# Maximum Likelihood Estimation to estimate the parameters a and b and Sigma
# The procedure is as follows: Start with initial guess for Sigma = I
# estimate a and b using the data (mle), then estimate Sigma using the residuals
# Calculate the distance between the estimated Sigma and the initial guess
# and use it to update the initial guess. Repeat until convergence.

function mle(params, data, Sigma, Mu)
    aa = params[1:9]  # First 9 parameters are a
    bb = params[10:12]  # Last 3 parameters are b
    # Sigma = reshape(params[13:end], 3, 3)  # Last 9 parameters are Sigma, reshape to 3x3 matrix
    # Calculate the log-likelihood
    log_likelihood = 0.0
    for i in 1:Nobs
        type = Int(data.type[i])
        what = aa[(type-1)*3+1:type*3] .+ data.exp[i] .* bb
        err_hat = collect(data[i, 1:3]) .- what
        log_likelihood += log(pdf(MvNormal(Mu, Sigma), err_hat))
    end
    return -log_likelihood  # Return negative log-likelihood for minimization
end  


i = 0
Sigma = [1.0 0.0 0.0; 
        0.0 1.0 0.0; 
        0.0 0.0 1.0]  # Initial guess for Sigma, identity matrix
params = vcat(0.05*ones(9), 0.1*ones(3))  # Initial guess for a, b, and Sigma

norm_diff = 1e6
param_diff = 1e6
while norm_diff > 1e-6 || param_diff > 1e-6
    i += 1
    optim_f = params -> mle(params, data, Sigma, Mu)
    # Run the optimization
    result = optimize(optim_f, params, NelderMead())
    # Extract the estimated parameters
    estimated_params = result.minimizer
    a_est = estimated_params[1:9]
    b_est = estimated_params[10:12]
    
    # Calculate the residuals at the estimated parameters
    residuals = zeros(Nobs, 3)
    for i in 1:Nobs
        type = Int(data.type[i])
        what = a_est[(type-1)*3+1:type*3] .+ data.exp[i] * b_est[type]
        residuals[i, :] = collect(data[i, 1:3]) .- what
    end
    
    # Estimate Sigma from the residuals
    Sigma_est = cov(residuals)
    
    # Calculate the norm of the difference between params and sigma in each iteration
    norm_diff = maximum(abs.(Sigma_est - Sigma))
    param_diff = maximum(abs.(estimated_params - params))
    @show norm_diff, param_diff, i
    # Update the initial guess for Sigma
    Sigma .= Sigma_est
    params .= estimated_params
end
