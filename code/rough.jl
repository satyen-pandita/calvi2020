include("mle_3.jl");

using BenchmarkTools
p = 0.01*ones(18*26+24);
Sigma = Matrix{Float64}(Matrix(I(9)));


@btime begin
    for i in 1:10
        optim_func(p, Sigma)
    end
end