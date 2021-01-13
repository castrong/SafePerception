using LinearAlgebra
using JuMP
using GLPK
using Distributions
using LazySets

include("nnet_functions.jl")

function inputPolytopeAroundPoint(nnet::NNet, x₀)
    # First, get modified weights that contain biases
    W̃ = get_mod_weights(nnet)
    AP = get_activation_pattern(x₀, W̃)
    A, b = get_constraints(AP, W̃)
    return HPolytope(A, b)
end

function approxOptima(nnet::NNet, x₀, radius; num_samples = 100, find_max = true)
    # First, get modified weights that contain biases
    W̃ = get_mod_weights(nnet)

    # Next, generate samples to obtain vector of activation patterns
    lb = max.(x₀ .- radius * ones(length(x₀)), 0) # Lower bounds of input region
    ub = min.(x₀ .+ radius * ones(length(x₀)), 1) # Upper bounds of input region
    dist = Uniform.(lb, ub)
    
    # start = time()
    APs = []
    for _ = 1:num_samples
        sample = rand.(dist)
        AP = get_activation_pattern(sample, W̃)
        !(AP in APs) ? push!(APs, AP) : nothing
    end
    # println(time() - start)

    # Solve an LP for each activation pattern
    # start = time()
    vals = zeros(length(APs))
    for i = 1:length(APs)
        A, b = get_constraints(APs[i], W̃)
        C, d = affine_map(APs[i], W̃)
        vals[i] = solve_LP(A, b, C, d, x₀, radius = radius, find_max = find_max)
    end
    # println(time() - start)

    # Return the optimum
    return find_max ? maximum(vals) : minimum(vals)
end

function get_mod_weights(nnet::NNet)
    W̃ = []
    
    W = nnet.weights[1]
    b = nnet.biases[1]
    for i = 1:length(W)
        w̃ = [W[i] b[i]; zeros(1, size(W[i], 2)) 1]
        push!(W̃, w̃)
    end
    return W̃
end

function evaluate_network_mod_weights(x, W̃)
    x̃ = [x; 1]
    output = x̃
    for i = 1:length(W̃) - 1
        ẑ = W̃[i] * output
        output = max.(ẑ, 0)
    end
    output = W̃[end] * output
    return output[1:end-1]
end

function get_activation_pattern(x, W̃)
    x̃ = [x; 1]
    output = x̃

    AP  = []
    for i = 1:length(W̃) - 1
        ẑ = W̃[i] * output
        output = max.(ẑ, 0)
        push!(AP, ẑ .> 0)
    end
    # push!(AP, trues(length(W̃[end])))
    return AP
end

# Return affine map Cx + d
function affine_map(AP, W̃)
    x_dim = size(W̃[1], 2)
    y_dim = size(W̃[end], 1) - 1
    
    W̃ᶜ = []
    for i = 1:length(AP)
        push!(W̃ᶜ, diagm(AP[i]) * W̃[i])
    end

    am = W̃ᶜ[1]
    for i = 2:length(AP)
        am = W̃ᶜ[i] * am
    end
    am = W̃[end] * am

    C = am[1:y_dim, 1:x_dim-1]
    d = am[1:y_dim, end]

    return C, d
end

# Return Ax ≤ b
function get_constraints(AP, W̃)
    AP_corrections = diagm(1 .- 2 .* AP[1]) * W̃[1]
    constraints = AP_corrections[1:end-1, :]

    lin_map = diagm(AP[1]) * W̃[1]
    for i = 2:length(AP)
        curr_prod = W̃[i] * lin_map
        AP_corrections = diagm(1 .- 2 .* AP[i]) * curr_prod
        constraints = vcat(constraints, AP_corrections[1:end-1, :])
        lin_map = diagm(AP[i]) * W̃[i] * lin_map
    end

    A = constraints[:, 1:end-1]
    b = -constraints[:, end]

    return A, b
end

function remove_degenerate(A, b)
    non_degenerate_inds = []
    for i = 1:size(A, 1)
        if any(A[i,:] .> 0)
            push!(non_degenerate_inds, i)
        end
    end

    return A[non_degenerate_inds, :], b[non_degenerate_inds, :]
end

function solve_LP(A, b, C, d, x₀; radius = 0.02, find_max = true)
    model = Model(GLPK.Optimizer)

    lb = max.(x₀ .- radius * ones(length(x₀)), 0)
    ub = min.(x₀ .+ radius * ones(length(x₀)), 1)

    @variable(model, x[1:size(A, 2)])
    @constraint(model, con[i in 1:size(A, 1)], dot(A[i,:], x) ≤ b[i])
    @constraint(model, lb .≤ x .≤ ub)
    if find_max
        @objective(model, Max, 0.015dot(C[1,:], x) + 0.008dot(C[2,:], x) + 0.015d[1] + 0.008d[2])
    else
        @objective(model, Min, 0.015dot(C[1,:], x) + 0.008dot(C[2,:], x) + 0.015d[1] + 0.008d[2])
    end

    optimize!(model)
    return objective_value(model)
end

########### Joe's function just for comparison testing
function get_constraints_Joe(weights, state, num_neurons)
	L = length(weights)
	# Initialize necessary data structures #
	idx2repeat = Dict{Int64,Vector{Int64}}() # Dict from indices of A to indices of A that define the same constraint (including itself)
	zerows = Vector{Int64}() # indices that correspond to degenerate constraints
	A = Matrix{Float64}(undef, num_neurons, size(weights[1],2)) # constraint matrix. A[:,1:end-1]x ≤ -A[:,end]
	lin_map = I
	# build constraint matrix #
	i = 1
	for layer in 1:L-1
		output = weights[layer]*lin_map
		for neuron in 1:length(state[layer])
			A[i,:] = (1-2*state[layer][neuron])*output[neuron,:]
			if !isapprox(A[i,1:end-1], zeros(size(A,2)-1), atol=1e-10) # check nonzero.
				#A[i,:] = normalize_row(A[i,:])
			else
				push!(zerows, i)
			end
			i += 1
		end
		lin_map = diagm(0 => state[layer])*weights[layer]*lin_map
	end

	return A[:,1:end-1], -A[:,end], idx2repeat, zerows
end