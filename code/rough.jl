include("mle_3.jl");

using BenchmarkTools
p = 0.01*ones(18*26+24);
Sigma = Matrix{Float64}(Matrix(I(9)));

@btime Z = compute(αm1, αm2, αm3, αf1, αf2, αf3, αc1, αc2, αc3, βm, βf, βc, ηm1, ηm2, ηm3, ηf1, ηf2, ηf3, DATA, NX);
@btime errs = gen_errors(αm1, αm2, αm3, αf1, αf2, αf3, αc1, αc2, αc3, βm, βf, βc, ηm1, ηm2, ηm3, ηf1, ηf2, ηf3, DATA, NX);
@btime ll = log_likelihood(errs, Sigma);

@btime optim_func(p, Sigma);