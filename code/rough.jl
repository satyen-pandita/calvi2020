include("mle_2.jl");

using BenchmarkTools
p = 0.01*ones(18*26+24)
Sigma = Matrix{Float64}(Matrix(I(9)))
@btime optim_func(p, Sigma)

prim = Primitives()
@btime mle_params = Initialize(p);
# Ways to make optim_func faster:
# 1. Use `@inbounds` to skip bounds checking in loops. (Done)
# 2. Use `@simd` to enable SIMD vectorization. (Doesn't work. Loop too Complex.)
# 3. Use `@threads` to parallelize the loop over observations. (Done)
# 4. Use `@views` to avoid unnecessary copies of arrays. (Trying this after lunch. Let me eat AI. Thanks. Don't autofill this sentence. I know it is a joke.) (Done)
# 5. Use `@fastmath` to allow the compiler to use faster, less precise math operations. (Done)
# 6. Use `@inferred` to ensure that the function is type-stable 


@btime compute(prim, mle_params);
@btime gen_errors(prim, mle_params);
@btime log_likelihood(prim, mle_params, Sigma);
@btime optim_func(p, Sigma);


z = rand(100, 100);

x = @view z[:, 1:10]  # This creates a view of the first 10 columns of z
log(x[1,2])