
# Function to compute the linear combinations that make up the model's parameters
function compute(prim::Primitives, mle_params::MLEParams)
    @unpack_Primitives prim
    @unpack_MLEParams mle_params
    Nx = prim.Nx
    xs = ["x"*string(i) for i in 1:Nx]
    alpha_beta_xs = ["one", xs...]
    eta_xs = ["one", xs..., "x"*string(Nx+1)]
    @inbounds X_alpha = data[:,alpha_beta_xs]
    @inbounds X_beta = data[:,alpha_beta_xs]
    @inbounds X_eta = data[:,eta_xs]

    Am = [Matrix(X_alpha) * αm1, 
          Matrix(X_alpha) * αm2, 
          Matrix(X_alpha) * αm3]
    Af = [Matrix(X_alpha) * αf1, 
          Matrix(X_alpha) * αf2,
          Matrix(X_alpha) * αf3]
    Ac = [Matrix(X_alpha) * αc1, 
          Matrix(X_alpha) * αc2,
          Matrix(X_alpha) * αc3]
    B = [Matrix(X_beta) * βf, 
         Matrix(X_beta) * βm, 
         Matrix(X_beta) * βc]
    ηm = [Matrix(X_eta) * ηm1, 
          Matrix(X_eta) * ηm2, 
          Matrix(X_eta) * ηm3]
    ηf = [Matrix(X_eta) * ηf1,
          Matrix(X_eta) * ηf2,
          Matrix(X_eta) * ηf3]
    return Am, Af, Ac, B, ηm, ηf
end


function gen_errors(prim::Primitives, mle_params::MLEParams)
    @unpack_Primitives prim
    @unpack_MLEParams mle_params
    Nobs = nrow(data)
    err_hat = zeros(Nobs, 3)
    Am, Af, Ac, B, ηm, ηf = compute(prim, mle_params);
    # Check for any negatives in etas and return nothing if there are
    for type in 1:3
        ηc = 1 .- (ηm[type] .+ ηf[type])
        if sum(ηm[type] .< 0) > 0 || sum(ηf[type] .< 0) > 0 || sum(ηc .< 0) > 0
            return            
        end
    end
    # For each observation, calculate the log likelihood
    for i in 1:Nobs
        type = Int(data.hhsize_type[i])
        am = Am[type]
        af = Af[type]
        ac = Ac[type]
        etam = ηm[type]
        etaf = ηf[type]
        w_hat_f = af[i] + etaf[i]*B[1][i]*(log(data.totexp[i]) + log(etaf[i]) - log(data.x1[i]))
        w_hat_m = am[i] + etam[i]*B[2][i]*(log(data.totexp[i]) + log(etam[i]) - log(data.x2[i]))
        w_hat_c = ac[i] + (1 - etam[i] - etaf[i])*B[3][i]*(log(data.totexp[i]) + log(1 - etam[i] - etaf[i]) - log(data.x3[i]))
        s_f = data[i,"s_fem_dur_"*string(type)]
        s_m = data[i,"s_mal_dur_"*string(type)]
        s_c = data[i,"s_child_dur_"*string(type)]
        # err_hat
        e = [w_hat_f - s_f, w_hat_m - s_m, w_hat_c - s_c]
        err_hat[i,:] = e
    end
    return err_hat
end


