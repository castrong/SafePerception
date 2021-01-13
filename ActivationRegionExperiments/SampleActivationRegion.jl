using NPZ
using Plots
using Colors

include("./RPMFunctions.jl")

# Load in the network
network_file = "/Users/cstrong/Desktop/Stanford/Research/SafePerception/ActivationRegionExperiments/AutoTaxiNetworks/AutoTaxi_32Relus_200Epochs_OneOutput.nnet"
network = read_network(network_file)

# Load in an input
input_file = "/Users/cstrong/Desktop/Stanford/Research/SafePerception/ActivationRegionExperiments/AutoTaxiInputs/AutoTaxi_12345_transposed.npy"
input = vec(npzread(input_file))
n = length(input)
lower = 0.0
upper = 1.0

# Find the activation pattern for this input and its associated input polytope
polytope = inputPolytopeAroundPoint(network, input)
polytope = intersection(polytope, Hyperrectangle(low=lower .* ones(n), high=upper .* ones(n)))

# Sample from this polytope using lazy sets (I believe rejection sampling)
num_samples = 2
println("Sampling")
samples = [reshape(sample(polytope), (16,8)) for i = 1:num_samples]

# Normalize then Visualize the samples
[Gray.(samples[i]) for i = 1:length(samples)]