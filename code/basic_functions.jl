
# Function to compute the linear combinations that make up the model's parameters
function compute(αm1, αm2, αm3, αf1, αf2, αf3, 
                 αc1, αc2, αc3, βm, βf, βc, 
                 ηm1, ηm2, ηm3, ηf1, ηf2, ηf3)

    Am = [X_ALPHA_BETA_MAT * αm1, X_ALPHA_BETA_MAT * αm2, X_ALPHA_BETA_MAT * αm3]
    Af = [X_ALPHA_BETA_MAT * αf1, X_ALPHA_BETA_MAT * αf2, X_ALPHA_BETA_MAT * αf3]
    Ac = [X_ALPHA_BETA_MAT * αc1, X_ALPHA_BETA_MAT * αc2, X_ALPHA_BETA_MAT * αc3]
    B = [X_ALPHA_BETA_MAT * βf, X_ALPHA_BETA_MAT * βm, X_ALPHA_BETA_MAT * βc]
    ηm = [X_ETA_MAT * ηm1, X_ETA_MAT * ηm2, X_ETA_MAT * ηm3]
    ηf = [X_ETA_MAT * ηf1, X_ETA_MAT * ηf2, X_ETA_MAT * ηf3]
    
    return Am, Af, Ac, B, ηm, ηf
end

# Function to generate errors based on the model's parameters and the data
# These errors are used to compute the log likelihood
function gen_errors(αm1, αm2, αm3, αf1, αf2, αf3, 
                    αc1, αc2, αc3, βm, βf, βc, 
                    ηm1, ηm2, ηm3, ηf1, ηf2, ηf3)
    Nobs = nrow(DATA)
    err_hat = zeros(Nobs, 3)
    Am, Af, Ac, B, ηm, ηf = compute(αm1, αm2, αm3, αf1, αf2, αf3, 
                                    αc1, αc2, αc3, βm, βf, βc, 
                                    ηm1, ηm2, ηm3, ηf1, ηf2, ηf3)
    # Check for any negatives in etas and return nothing if there are
    for type in 1:3
        ηc = 1 .- (ηm[type] .+ ηf[type])
        if sum(ηm[type] .< 0) > 0 || sum(ηf[type] .< 0) > 0 || sum(ηc .< 0) > 0
            return nothing            
        end
    end
    # For each observation, calculate the errors
    for type in 1:3
        Threads.@threads for i in TYPE_INDICES[type]
            am = Am[type]
            af = Af[type]
            ac = Ac[type]
            etam = ηm[type]
            etaf = ηf[type]
            w_hat_f = af[i] + etaf[i]*B[1][i]*(log(DATA.totexp[i]) + log(etaf[i]) - log(DATA.x1[i]))
            w_hat_m = am[i] + etam[i]*B[2][i]*(log(DATA.totexp[i]) + log(etam[i]) - log(DATA.x2[i]))
            w_hat_c = ac[i] + (1 - etam[i] - etaf[i])*B[3][i]*(log(DATA.totexp[i]) + log(1 - etam[i] - etaf[i]) - log(DATA.x3[i]))
            s_f = DATA[i,"s_fem_dur_"*string(type)]
            s_m = DATA[i,"s_mal_dur_"*string(type)]
            s_c = DATA[i,"s_child_dur_"*string(type)]
            # err_hat
            e = [w_hat_f - s_f, w_hat_m - s_m, w_hat_c - s_c]
            err_hat[i,:] = e    
        end
    end

    return err_hat
end   