using Flux 
using Flux: logitbinarycrossentropy, glorot_uniform
using Flux.Data: DataLoader
using Flux.Optimise: train!
using Parameters
using Random
using NeuralVerification
using NeuralVerification: ReLU, Layer, Id
using LazySets 

include("./RPMFunctions.jl")
include("./TilingHelpers.jl")
include("./flux2nnet.jl")
include("./NetworkSimplification.jl")

function get_counts(network, X; print_freq = 5000)
    counts = Dict()
    for i = 1:size(X, 2)
        i % print_freq == 0 && println("Getting counts, ", round(i / size(X, 2) * 100, digits=1), "% done")
        # Get the activation pattern
        x = X[:, i]
        AP = get_activation_pattern(network, x)
    
        # Update the dictionary
        if !haskey(counts, AP)
            counts[AP] = 0
        end
        counts[AP] = counts[AP] + 1
    end
    return counts
end

function approximate_number_regions(network, cell, num_samples)
    xs = LazySets.sample(cell, num_samples)
    X = zeros(length(xs[1]), num_samples)
    # Put them into a matrix where each column is a sample 
    for (i, x) in enumerate(xs) 
        X[:, i] = x
    end
    # put into a matrix 
    counts = get_counts(network, X)
    return length(counts)
end

function convert_flux_to_NNet(network)
   # First convert it to the NNet format from the utils here 
    # we do this by writing it as a .nnet then reading that in 
    mktemp() do path, io
        temp_file = path*".nnet"
        n_inputs = size(network[1].W, 2)
        n_outputs = size(network[end].W, 1)
        flux2nnet(temp_file, network, -9999999*ones(n_inputs), 99999999*ones(n_inputs), zeros(n_inputs), ones(n_inputs))
        converted_network = read_network(temp_file)
        rm(temp_file)
        return converted_network
    end
end

function approximate_number_regions_fluxmodel(network, cell, num_samples)
    converted_network = convert_flux_to_NNet(network)
    return approximate_number_regions(converted_network, cell, num_samples)
end

"""
    make_random_network(layer_sizes::Vector{Int}, [min_weight = -1.0], [max_weight = 1.0], [min_bias = -1.0], [max_bias = 1.0], [rng = 1.0])
    read_layer(output_dim::Int, f::IOStream, [act = ReLU()])
Generate a network with random weights and bias. The first layer is treated as the input.
The values for the weights and bias will be uniformly drawn from the range between min_weight
and max_weight and min_bias and max_bias respectively. The last layer will have an ID()
activation function and the rest will have ReLU() activation functions. Allow a random number
generator(rng) to be passed in. This allows for seeded random network generation.
"""
function make_random_network(layer_sizes::Vector{Int}, min_weight = -1.0, max_weight = 1.0, min_bias = -1.0, max_bias = 1.0, rng=MersenneTwister())
    # Create each layer based on the layer_size
    layers = []
    for index in 1:(length(layer_sizes)-1)
        cur_size = layer_sizes[index]
        next_size = layer_sizes[index+1]
        # Use Id activation for the last layer - otherwise use ReLU activation
        if index == (length(layer_sizes)-1)
            cur_activation = Id()
        else
            cur_activation = ReLU()
        end

        # Dimension: num_out x num_in
        cur_weights = min_weight .+ (max_weight - min_weight) * rand(rng, Float64, (next_size, cur_size))
        cur_weights = reshape(cur_weights, (next_size, cur_size)) # for edge case where 1 dimension is equal to 1 this keeps it from being a 1-d vector

        # Dimension: num_out x 1
        cur_bias = min_bias .+ (max_bias - min_bias) * rand(rng, Float64, (next_size))
        push!(layers, Layer(cur_weights, cur_bias, cur_activation))
    end

    nv_network = Network(layers)

    # Now, convert to our NNet format that the utils are written with 
    mktemp() do path, io
        temp_file = path*".nnet"
        println("path: ", path)
        NeuralVerification.write_nnet(temp_file, nv_network)
        network = read_network(temp_file)
        rm(temp_file)
        return network
    end
end

# Run an experiment where you generate a bunch of networks with 
# different number of nodes and see how the number of regions changes 
function run_num_nodes_exp()
    cell = Hyperrectangle(low=[-0.5, -0.5], high=[0.5, 0.5])
    num_samples = 1000000
    input_dim = 2
    largest_layer_sizes = [input_dim, 16, 16, 16, 16, 32, 32, 32, 16, 16, 16] #[input_dim, 3, 4]
    # add layers in one by one. repeat each size num_repeats times 
    num_repeats = 7
    region_counts = zeros((length(largest_layer_sizes)-1) * num_repeats)
    node_nums = zeros((length(largest_layer_sizes)-1) * num_repeats)

    for i = 2:length(largest_layer_sizes)
        for j = 1:num_repeats
            println("Num nodes exp layer: ", i)
            current_layers = largest_layer_sizes[1:i]
            network = make_random_network(current_layers)
            num_regions = approximate_number_regions(network, cell, num_samples)
            region_counts[(i-2)*num_repeats + j] = num_regions
            node_nums[(i-2)*num_repeats + j] = sum(current_layers)
        end
    end
    
    #expected = [nodes^input_dim/factorial(input_dim) for nodes in node_nums]

    # The first element of node_nums and region_counts will not have actually been run 
    scatter(node_nums, region_counts, title="Activation Regions vs. Number of Nodes", xlabel="Number of Nodes", ylabel="Activation Regions")
    #plot!(node_nums, expected, label="Expected Relationship")
    savefig("./ActivationRegionExperiments/EE207_plots/num_node_exp.png")
    println("node nums: ", node_nums)
    println("region counts: ", region_counts)
end


function run_num_input_nodes_exp()
    num_samples = 100000

    input_sizes = [1, 2, 3, 4, 5, 6]
    rest_of_layers = [3, 16, 32, 32, 32, 16] #[16, 32, 32, 16]

    num_repeats = 5

    # Count the number of regions with each network 
    region_counts = zeros(length(input_sizes) * num_repeats)
    input_sizes_for_plot = zeros(length(input_sizes) * num_repeats)

    for (i, input_size) in enumerate(input_sizes)
        for j = 1:num_repeats
            # Cell side lengths of 1 to maintain a 
            cell = Hyperrectangle(low=-0.5 * ones(input_size), high=0.5 * ones(input_size))

            current_layers = [input_size; rest_of_layers]
            network = make_random_network(current_layers)
            num_regions = approximate_number_regions(network, cell, num_samples)
            region_counts[(i-1)*num_repeats + j] = num_regions
            input_sizes_for_plot[(i-1)*num_repeats + j] = input_size
        end
    end

    scatter(input_sizes_for_plot, region_counts, title="Activation Regions vs. Number of Inputs", xlabel="Number of Inputs", ylabel="Activation Regions", legend=:none)
    savefig("./ActivationRegionExperiments/EE207_plots/num_inputs_exp.png")
    println("inputs: ", input_sizes_for_plot)
    println("region counts: ", region_counts)
end


function build_model(layer_sizes, act; init=glorot_uniform)
    println("Building model")
    # ReLU except last layer softmax
    layers = Any[Dense(layer_sizes[i], layer_sizes[i+1], act) for i = 1:length(layer_sizes) - 2]
    push!(layers, Dense(layer_sizes[end-1], layer_sizes[end]))
    println("Layers: ", layers)
    return Chain(layers...)
end


# function get_network_regions(network::NNet, X)
#    # Get the counts for each activation pattern
#    counts = get_counts(network, X)
#    # We don't actually need to sort it in this fcn because we're 
#    # not trimming any based on their counts 
#    ap_sorted, counts_sorted = sort_dict_by_value(counts)
#    regions = Array{AffinePolytopeRegion, 1}(undef, length(ap_sorted))
#    for i = 1:length(regions)
#        polytope, A, b = polytopeAndMapFromAP(network, ap_sorted[end-i+1])
#        regions[i] = AffinePolytopeRegion(intersection(polytope, domain), A, b) # intersect the polytope with the domain here
#    end
#    return regions 
# end


# # Input: a network and a dataset X where each column is an input to the network
# # Output: A list of affine polytope regions for a network that you'd like to use to approximate it
# function get_and_visualize_network_regions(network::NNet, X, domain, filename)
#     regions = get_network_regions(network, X)
#     centers = polytope_centers(regions)
#     visualize_regions(network, X, length(regions), domain; regions=regions, centers=centers, plot_points = false, save_filename=filename)
# end


# spherical decision boundary 
get_label(x) = norm(x) >= 0.5

# X is a dataset with each column being an input 
# y is a *row* vector of binary labels 

# Run it once 
function run_training_exp(random_labels=false)
    # Define parameters for your experiment 
    n_input_dims = 2
    n_samples = 1000
    X = (rand(MersenneTwister(0), n_input_dims, n_samples) .- 0.5) .* 2.0 # have the range be -1 --> 1
    y = random_labels ? rand(MersenneTwister(1), n_samples) : [get_label(x) for x in eachcol(X)] # have a circular decision boundary    # 
    layer_sizes = [n_input_dims, 10, 10, 10, 10, 1]

    num_epochs = 100

    # Parameters for counting the activation regions
    num_samples_counting = 100000
    # cover the full range data is in 
    cell = Hyperrectangle(low=-1.0 * ones(n_input_dims), high=1.0 * ones(n_input_dims))
    # have a matrix of Xs on hand of that size to pass to the visualization function 
    # this is a sort of awkward workaround 
    xs = LazySets.sample(cell, num_samples_counting)
    X_largesample = zeros(length(xs[1]), num_samples_counting)
    # Put them into a matrix where each column is a sample 
    for (i, x) in enumerate(xs) 
        X_largesample[:, i] = x
    end

    # Build your model and setup the cross entropy loss 
    model = build_model(layer_sizes, relu)

    # visualize the untrained activation regions 
    # interestingly it looks like biases are initialized to 0 when we create a Dense layer 
    # and it's not uncommon for that to be the case 
    # https://stackoverflow.com/questions/44883861/initial-bias-values-for-a-neural-network#:~:text=Initializing%20the%20biases.,random%20numbers%20in%20the%20weights.
    initial_num_regions = approximate_number_regions_fluxmodel(model, cell, num_samples_counting)
    converted_model = convert_flux_to_NNet(model)
    #visualize_regions(converted_model, X_largesample, Int(initial_num_regions), cell; save_filename=string("./ActivationRegionExperiments/EE207_plots/activation_regions_untrained_random=", random_labels, ".png"))


    # Probably inefficient looping but I had trouble batching this loss 
    loss(x, label) = begin 
        total = 0
        for i = 1:size(x, 2)
            total = total + logitbinarycrossentropy(model(x[:, i])[1], label[i])
        end
        return total
    end 


    # Batch the data, and configure your optimizer 
    data = DataLoader(X, y, batchsize=128) 
    opt = ADAM()

    # Pull out the params, then train.
    ps = Flux.params(model)

    epoch_indices = collect(1:num_epochs)
    region_nums = zeros(num_epochs)

    losses = zeros(num_epochs)
    for i = 1:num_epochs
        println("Num epoch: ", i)
        losses[i] = loss(X, y)
        println("    Loss at start: ", losses[i])
        # Trains for a single epoch? 
        train!(loss, ps, data, opt)

        region_nums[i] = approximate_number_regions_fluxmodel(model, cell, num_samples_counting)
    end

    plot(epoch_indices, region_nums, title=string("Activation Regions vs. Epoch", random_labels ? " With Random Labels" : ""), xlabel="Epoch", ylabel="Activation Regions", legend=:none)
    savefig(string("./ActivationRegionExperiments/EE207_plots/training_exp_randomlables=", random_labels, ".png"))
    println("use random: ", random_labels)
    println("epoch_indices = ", epoch_indices)
    println("region_nums = ", region_nums)

    # plot the loss too 
    plot(epoch_indices, losses, title=string("Loss vs. Epoch", random_labels ? " With Random Labels" : ""), xlabel="Epoch", ylabel="Loss")
    savefig(string("./ActivationRegionExperiments/EE207_plots/training_exp_loss_randomlables=", random_labels, ".png"))

    # Visualize the tiling it's learned 
    converted_model = convert_flux_to_NNet(model)
    #visualize_regions(converted_model, X_largesample, Int(region_nums[end]), cell; save_filename=string("./ActivationRegionExperiments/EE207_plots/activation_regions_trained_random=", random_labels, ".png"))
    
    #get_and_visualize_network_regions(converted_model, X, cell, string("./EE207_plots/activation_regions_trained_random=", random_labels, ".png"))

    return model 
end

#run_num_nodes_exp()
run_num_input_nodes_exp()
#model_circle = run_training_exp(false)
#model_random = run_training_exp(true)


# Results:

### Num node experiment 
# cell = Hyperrectangle(low=[-0.5, -0.5], high=[0.5, 0.5])
# num_samples = 1000000
# input_dim = 2
# largest_layer_sizes = [input_dim, 16, 16, 16, 16, 32, 32, 32, 16, 16, 16] #[input_dim, 3, 4]
# # add layers in one by one. repeat each size num_repeats times 
# num_repeats = 7
# node_nums = [18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 162.0, 162.0, 162.0, 162.0, 162.0, 162.0, 162.0, 178.0, 178.0, 178.0, 178.0, 178.0, 178.0, 178.0, 194.0, 194.0, 194.0, 194.0, 194.0, 194.0, 194.0, 210.0, 210.0, 210.0, 210.0, 210.0, 210.0, 210.0]
# region_counts = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 13.0, 18.0, 26.0, 36.0, 12.0, 13.0, 40.0, 62.0, 75.0, 26.0, 27.0, 54.0, 76.0, 97.0, 69.0, 104.0, 253.0, 106.0, 75.0, 94.0, 47.0, 210.0, 299.0, 381.0, 55.0, 448.0, 123.0, 161.0, 285.0, 307.0, 561.0, 266.0, 247.0, 431.0, 253.0, 886.0, 93.0, 511.0, 314.0, 660.0, 391.0, 812.0, 872.0, 1576.0, 852.0, 684.0, 874.0, 165.0, 691.0, 338.0, 1896.0, 571.0, 474.0, 571.0, 906.0, 564.0, 1329.0, 1203.0, 1217.0, 1317.0, 831.0, 307.0, 938.0]


### Input nodes experiment 
# num_samples = 100000
# input_sizes = [1, 2, 3, 4, 5, 6]
# rest_of_layers = [3, 16, 32, 32, 32, 16] #[16, 32, 32, 16]
# num_repeats = 5

# input_sizes_for_plot = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]
# region_counts = [13.0, 1.0, 9.0, 17.0, 15.0, 16.0, 483.0, 186.0, 131.0, 70.0, 61.0, 233.0, 1423.0, 32.0, 154.0, 98.0, 133.0, 1100.0, 602.0, 1907.0, 1105.0, 4861.0, 1881.0, 340.0, 561.0, 215.0, 1371.0, 124.0, 359.0, 2931.0]


### Training experiment
# n_input_dims = 2
# n_samples = 1000
# X = (rand(MersenneTwister(0), n_input_dims, n_samples) .- 0.5) .* 2.0 # have the range be -1 --> 1
# y = random_labels ? rand(MersenneTwister(1), n_samples) : [get_label(x) for x in eachcol(X)] # have a circular decision boundary    # 
# layer_sizes = [n_input_dims, 10, 10, 10, 10, 1]

# num_epochs = 100

# # Parameters for counting the activation regions
# num_samples_counting = 100000
# # cover the full range data is in 
# cell = Hyperrectangle(low=-1.0 * ones(n_input_dims), high=1.0 * ones(n_input_dims))

# For circle classification:
# oops lost them ah welll 

# For random: 
# epoch_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
# region_nums = [408.0, 471.0, 476.0, 459.0, 481.0, 486.0, 492.0, 490.0, 474.0, 469.0, 463.0, 480.0, 483.0, 470.0, 459.0, 470.0, 464.0, 469.0, 455.0, 467.0, 474.0, 470.0, 461.0, 450.0, 459.0, 463.0, 448.0, 455.0, 453.0, 443.0, 484.0, 467.0, 466.0, 472.0, 473.0, 479.0, 474.0, 462.0, 471.0, 485.0, 454.0, 469.0, 483.0, 471.0, 487.0, 496.0, 497.0, 508.0, 523.0, 507.0, 509.0, 528.0, 524.0, 520.0, 516.0, 524.0, 534.0, 533.0, 540.0, 527.0, 525.0, 535.0, 532.0, 532.0, 510.0, 511.0, 520.0, 505.0, 511.0, 505.0, 516.0, 509.0, 527.0, 523.0, 522.0, 528.0, 536.0, 533.0, 532.0, 532.0, 529.0, 535.0, 541.0, 534.0, 529.0, 533.0, 528.0, 542.0, 540.0, 533.0, 523.0, 549.0, 525.0, 530.0, 530.0, 531.0, 538.0, 545.0, 558.0, 548.0]
