using Parameters
using Flux
using DelimitedFiles
using Images
using TestImages
using Random

# Visualization packages
using Reel
Reel.extension(m::MIME"image/svg+xml") = "svg"
Reel.set_output_type("gif") # may be necessary for use in IJulia

include("./RPMFunctions.jl")
include("./TilingHelpers.jl")
include("./flux2nnet.jl")

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

# Take in a dictionary, return a list of the keys and a list
# of the values, sorted by the value
# Input: A dictionary
# Output: a list of the keys and a list of the values sorted by value from lowest to highest
function sort_dict_by_value(dict)
    # Pull out the keys and values from the dictionary
    d_keys = collect(keys(dict))
    d_values = [dict[key] for key in d_keys]
    # Get the index permuation based on sorting the values valeus
    p = sortperm(d_values)
    # Update the keys and values to the new order
    d_values .= d_values[p]
    d_keys .= d_keys[p]
    return d_keys, d_values
end

# Input: a network and a dataset X where each column is an input to the network
# Output: A list of affine polytope regions for a network that you'd like to use to approximate it
function network_to_affine_polytope_regions(network::NNet, X, num_regions, domain)
    # Get the counts for each activation pattern
    counts = get_counts(network, X)
    ap_sorted, counts_sorted = sort_dict_by_value(counts)
    # Return the top num_regions 
    regions = Array{AffinePolytopeRegion, 1}(undef, num_regions)
    for i = 1:num_regions
        polytope, A, b = polytopeAndMapFromAP(network, ap_sorted[end-i+1])
        regions[i] = AffinePolytopeRegion(intersection(polytope, domain), A, b) # intersect the polytope with the domain here
    end
    return regions 
end
# Fill a domain around your list of affine polytopes
# Input: a list of polytope regions along with their affine mapping and the domain you want to fill
# Output: a list of AffinePolytopeRegions that has tiled that domain 
function tile_affine_polytope_regions(affine_polytope_regions, domain)
    # Get the polytopes from your regions
    polytopes = [region.domain for region in affine_polytope_regions]
    # Tile the region using those polytopes 
    expanded_polytopes = polytopes_to_decomposition(polytopes, domain)
    # Remake the affine polytope regions with the new expanded polytopes
    return [AffinePolytopeRegion(polytope=expanded_polytopes[i], A=affine_polytope_regions[i].A, b=affine_polytope_regions[i].b) for i = 1:length(polytopes)]
end

# Input: the network to simplify and input data X where each column is an input x to the network (each row is a different feature)
# Output: a list of AffinePolytopeRegion objects which each has a polytope and the affine function within that polytope
function simplify_network_to_regions(network, X, domain; num_regions=20)
    regions = network_to_affine_polytope_regions(network, X, num_regions, domain)
    return tile_affine_polytope_regions(regions, domain)
end

# Input: the network to simplify and input data X where each column is an input x to the network (each row is a different feature)
# Output: a list of AffineAndPoint objects which give one representation of the network's behavior
function simplify_network_to_points(network, X, domain; num_regions=20)
    regions = network_to_affine_polytope_regions(network, X, num_regions, domain)
    centers = polytope_centers(regions)
    return [AffineAndPoint(point=centers[i], A=regions[i].A, b=regions[i].b) for i = 1:length(centers)]
end

function compute_output(regions::Vector{AffineAndPoint}, x)
end

function compute_output(regions::Vector{AffinePolytopeRegion}, x)
end

# Evaluate a simplified network which stores points and an affine mapping 
# for the region close to each point. This finds the closest point, 
# and then applies the respective affine mapping
function (simplified_network::Vector{AffineAndPoint})(x)
    best_dist = Inf
    best_A = nothing
    best_b = nothing
    for region in simplified_network
        dist = norm(region.point .- x)
        if dist < best_dist
            best_dist = dist
            best_A = region.A
            best_b = region.b
        end
    end
    return best_A * x + best_b
end

# Evaluate a simplified network which is represented by polytopes each of 
# which has an affine mapping associated with it. 
# if the regions in this representation correspond to the Voronoi decomposition
# of a simplified network given in the AffineAndPoint form then the 
# output should match. 
function (simplified_network::Vector{AffinePolytopeRegion})(x)
    for region in simplified_network
        if x ∈ region.domain
            return region.A * x + region.b
        end
    end
end

# Evaluate a network on a dataset of input output pairs
# Input: network, input data X (each column is a data point), lables Y,
#        and loss_fcn which must take in the arguments (y_hat, y) and output a scalar
# Output: the sum of the loss on each data point
function evaluate_network(network, X, Y, loss_fcn)
    loss = 0
    for i = 1:size(X, 2)
        loss += loss_fcn(network(X[:, i]), Y[:, i])
    end
    return loss
end

function image_from_network(network, height, width, means, stdevs; print_period = 100)
    arr = zeros(height, width)
    for i = 1:height
        i % print_period == 0 && println("Evaluating row: ", i)
        for j = 1:width
            normalized_x = (Float64.([i, j]) .- means) ./ stdevs
            arr[i, j] = network(normalized_x)[1]
        end
    end
    return Gray.(arr)
end

# Input: Take in a network, input matrix (each column is an input)
#        the number of regions to plot and the domain 
# Output: A plot saved to save_filename which includes num_regions most popular polytopes
function visualize_regions(network, X, num_regions, domain
                           ;regions = network_to_affine_polytope_regions(network, X_normalized, num_regions, domain),
                           centers = polytope_centers(regions),
                           save_filename="./temp.png", plot_points=true, num_plot_points = 1000)
    cur_plot = plot(domain, aspect_ratio=:equal)
    for i = 1:num_regions
        cur_plot = plot!(intersection(regions[i].domain, domain))
        scatter!([centers[i][1]], [centers[i][2]], markersize=2.0, legend=:none)
    end
    # Plot some points from your dataset on top 
    if plot_points
        shuffled_X = X[:, shuffle(1:end)]
        points = shuffled_X[:, 1:num_plot_points]
        scatter!(points[1, :], points[2, :], markersize=0.8)
    end
    savefig(save_filename)
    return cur_plot
end

function visualize_decomposition(regions::Vector{AffineAndPoint}, domain; save_filename="./temp_decomposition.png", plot_points=true, num_plot_points = 1000)
    polytopes = voronoi_decomposition([region.point for region in regions]; domain=domain)
    cur_plot = plot(domain, aspect_ratio=:equal)
    for polytope in polytopes
        cur_plot = plot!(polytope)
    end
    savefig(save_filename)
    return cur_plot
end

# Create a gif of the most popular regions showing up one by one
function popular_regions_gif(network, X, num_regions, domain
    ;regions = network_to_affine_polytope_regions(network, X_normalized, num_regions, domain)
    , save_filename="./temp.gif")
    
    frames = Frames(MIME("image/svg+xml"), fps=10)
    for i = 1:num_regions
        cur_plot = visualize_regions(network, X, i, domain; regions=regions, plot_points=false)
        push!(frames, cur_plot)
    end
    println("Saving gif")
    write(save_filename, frames)
end

# Load in the data and the normalization 
X = readdlm("./Data/test_image_X.csv", ',', Float64, '\n')
Y = readdlm("./Data/test_image_Y.csv", ',', Float64, '\n')
means = readdlm("./Data/test_image_means.csv", ',', Float64, '\n')
stdevs = readdlm("./Data/test_image_stdevs.csv", ',', Float64, '\n')
X_normalized = (X .- means) ./ stdevs

num_regions = 500

# Load in the network
network_file = "./ActivationRegionExperiments/Models/model_initial_test_(2, 32, 32, 1).bson"
#network_file = "./ActivationRegionExperiments/Models/model_initial_test_(2, 32, 64, 64, 128, 128, 128, 64, 64, 64, 64, 32, 1).bson"
#network_file = "./ActivationRegionExperiments/Models/model_initial_test_(2, 32, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 1024, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 64, 64, 32, 1).bson"
network = read_bson_network(network_file, [1.0, 1.0], [512.0, 512.0], means, stdevs)


# Compress the network and see how it differs
domain_low = vec(([1.0, 1.0] .- means)./stdevs)
domain_high = vec(([512.0; 512.0] .- means) ./stdevs)
domain = Hyperrectangle(low=domain_low, high=domain_high)
compressed_network = simplify_network_to_points(network, X_normalized, domain; num_regions=num_regions)

# Compare the loss between the original and compressed networks 
original_loss = evaluate_network(network, X_normalized, Y, (ŷ, y) -> (ŷ[1] - y[1])^2)
compressed_loss = evaluate_network(compressed_network, X_normalized, Y, (ŷ, y) -> (ŷ[1] - y[1])^2)

# Visualize what the network has learned and compare it to the compressed network
original_network_image = image_from_network(network, 512, 512, means, stdevs)
compressed_network_image = image_from_network(compressed_network, 512, 512, means, stdevs)
groundtruth = Gray.(testimage("mandrill"))
Images.save("./ActivationRegionExperiments/Plots/network_image.png", map(clamp01nan, original_network_image))
Images.save("./ActivationRegionExperiments/Plots/compressed_network_image.png", map(clamp01nan, compressed_network_image))
Images.save("./ActivationRegionExperiments/Plots/groundtruth_image.png", groundtruth)

# Visualize the num_regions most popular regions
visualize_decomposition(compressed_network, domain)

regions = network_to_affine_polytope_regions(network, X_normalized, 100, domain)
centers = polytope_centers(regions)
visualize_regions(network, X_normalized, num_regions, domain; regions=regions, centers=centers, plot_points = false)
#popular_regions_gif(network, X_normalized, 37, domain; regions=regions)

println("Original loss: ", original_loss)
println("Compressed loss: ", compressed_loss)