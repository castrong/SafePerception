using NPZ
using Plots
using Colors
using HDF5
using DataStructures
using Distributions

include("./RPMFunctions.jl")

# Load in the network
network_file = "./ActivationRegionExperiments/AutoTaxiNetworks/AutoTaxi_32Relus_200Epochs_OneOutput.nnet"
network = read_network(network_file)

# Loop through the training data, counting the naumber of times we see each activation region
SK_counts = Dict()
KJ_counts = Dict()

SK_train_filename = "./Data/SK_DownsampledTrainingData.h5"
KJ_train_filename = "./Data/KJ_DownsampledTrainingData.h5"

SK_images = h5read(SK_train_filename, "y_train") # Meant for learning to generate images so images are y_train
KJ_images = h5read(KJ_train_filename, "X_train") # Meant for learning control so images are X_train
all_images = cat(SK_images, KJ_images, dims=(3))

function get_counts(network, images; print_freq = 1000)
    counts = Dict()
    for i = 1:size(images, 3)
        i % print_freq == 0 && println("Image: ", i)
        # Get the activation pattern
        image = images[:, :, i]
        shaped_image = reshape(image, 16*8)
        AP = get_activation_pattern(network, shaped_image)
    
        # Update the dictionary
        if !haskey(counts, AP)
            counts[AP] = 0
        end
        counts[AP] = counts[AP] + 1
    end
    return counts
end

sk_counts = get_counts(network, SK_images)
kj_counts = get_counts(network, KJ_images)
all_counts = get_counts(network, all_images)

println("Num activations in sk: ", length(sk_counts), "   for ", size(SK_images, 3), " images")
println("Num activations in kj: ", length(kj_counts), "   for ", size(KJ_images, 3), " images")
println("Num activations total: ", length(all_counts), "  for ", size(all_images, 3), " images")

sk_count_vals = [v for (k, v) in sk_counts]
sk_count_keys = [k for (k, v) in sk_counts]
kj_count_vals = [v for (k, v) in kj_counts]
kj_count_keys = [k for (k, v) in kj_counts]
all_count_vals = [v for (k, v) in all_counts]
all_count_keys = [k for (k, v) in all_counts]


plot(accumulate(+, sort(all_count_vals, rev=true) ./ sum(all_count_vals)), title="Cumulative # images in activation regions (128 ReLU network)", xlabel="Activation region index", ylabel="Cumulative percent of images in most popular AP")
#plot(sort(all_count_vals))

# Find the number of images in SK that were in KJ
num_in_kj = 0
for (sk_k, sk_v) in sk_counts
    global num_in_kj
    if sk_k in keys(kj_counts)
        num_in_kj = num_in_kj + sk_v
    end
end
println(num_in_kj, " out of ", size(SK_images, 3), " (", round(num_in_kj/size(SK_images, 3)*100, digits=2), "%) images from SK have AP from KJ")


# Generate random images and see how often they're in the KJ activations
num_rand = 100000
rand_images = [rand(128) for i = 1:num_rand]
rand_in_kj = 0
rand_in_sk = 0
rand_in_all = 0
for image in rand_images
    global rand_in_kj, rand_in_sk, rand_in_all
    AP = get_activation_pattern(network, image)
    rand_in_kj = rand_in_kj + (haskey(kj_counts, AP) ? 1 : 0)
    rand_in_sk = rand_in_sk + (haskey(sk_counts, AP) ? 1 : 0)
    rand_in_all = rand_in_all + (haskey(all_counts, AP) ? 1 : 0)
end
println(rand_in_kj, " out of ", num_rand, " (", round(rand_in_kj/num_rand*100, digits=2), "%) images from rand in KJ")
println(rand_in_sk, " out of ", num_rand, " (", round(rand_in_sk/num_rand*100, digits=2), "%) images from rand in SK")
println(rand_in_all, " out of ", num_rand, " (", round(rand_in_all/num_rand*100, digits=2), "%) images from rand in all")

# Using these random images, see how likely they are to be in the top 100 most popular activation patterns


# Then we'll want to do the same thing but with activation values themselves