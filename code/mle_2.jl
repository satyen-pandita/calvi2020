include("initialize.jl")
include("basic_functions.jl")

function log_likelihood(prim::Primitives, mle_params::MLEParams, Sigma::Matrix{Float64})
    @unpack_Primitives prim
    @unpack_MLEParams mle_params
    err_hat = gen_errors(prim, mle_params)
    if isnothing(err_hat)
        return 1e100
    end
    Nobs = nrow(data)
    # Generate the Log likelihood array for the errors so as to use @simd
    # This is not necessary, but it can help with performance if the array is large
    # Mu is (0,0,0), Sigma is given as an argument 
    Mu = zeros(3) 
    LL_array = zeros(Nobs, 3)
    @inbounds Threads.@threads for i in 1:Nobs
        # For each observation, calculate the log likelihood
        type = Int(data.hhsize_type[i])
        # Get the 3x3 Sigma matrix for this type
        @views S = Sigma[(type-1)*3+1:type*3, (type-1)*3+1:type*3]
        @fastmath LL_array[i,type] = log(pdf(MvNormal(Mu, S), err_hat[i,:]))
    end

    ll = 0.0
    # For each observation, calculate the log likelihood
    @inbounds for i in 1:Nobs
        @fastmath ll += LL_array[i, Int(data.hhsize_type[i])]
    end
    return -ll
end

prim = Primitives();
function optim_func(params::Vector{Float64}, Sigma::Matrix{Float64})
    mle_params = Initialize(params);
    ll = log_likelihood(prim, mle_params, Sigma);
    return ll
end

function optim_loop()    
    Nx = 26
    param_guess = 0.01*ones(18*Nx+24)
    sigma_guess = Matrix{Float64}(Matrix(I(9)))
    sigma_diff = 1e6
    param_diff = 1e6
    i = 0
    while sigma_diff > 1e-6
        i += 1
        optim_f = params -> optim_func(params, sigma_guess)
        res = optimize(optim_f, param_guess, NelderMead(), Optim.Options(iterations=1000, show_trace=true, show_every=1))
        params_new = res.minimizer
        prim, mle_params = Initialize(params_new)
        err_hat = gen_errors(prim, mle_params)
        err_hat_1, err_hat_2, err_hat_3 = err_hat[prim.data.hhsize_type .== 1.0], err_hat[prim.data.hhsize_type .== 2.0], err_hat[prim.data.hhsize_type .== 3.0]
        sigma_1, sigma_2, sigma_3 = cov(err_hat_1), cov(err_hat_2), cov(err_hat_3)
        sigma_new = [sigma_1 zeros(3) zeros(3);
                    zeros(3) sigma_2 zeros(3);
                    zeros(3) zeros(3) sigma_3]
        param_diff = maximum(abs.(params_new .- param_guess))
        sigma_diff = maximum(abs.(sigma_new .- sigma_guess))
        sigma_guess = sigma_new
        param_guess = params_new 
        @show i, param_diff, sigma_diff
    end
    return param_guess, sigma_guess
end


# params, sigma = optim_loop();

# optim_f = params -> optim_func(params, init_sigma)

# res = optimize(optim_f, init_guess, NelderMead(), Optim.Options(iterations=1000, show_trace=true, show_every=1))

