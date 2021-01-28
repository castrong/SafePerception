using NPZ
using Plots
using Colors
using HDF5
using DataStructures

# Load in the network
network_file = "./ActivationRegionExperiments/AutoTaxiNetworks/AutoTaxi_128Relus_200Epochs_OneOutput.nnet"
network = read_network(network_file)

y_true = y_label[1, :] .* .015 + y_label[2, :] * .008
y_label = h5read(KJ_train_filename, "y_train") # Meant for learning control so images are X_train

y_hat = [evaluate_network(network, reshape(KJ_images[:, :, i], 16*8))[1] for i = 1:size(KJ_images, 3)]
y_hat_transpose = [evaluate_network(network, Array(reshape(transpose(KJ_images[:, :, i]), 16*8)))[1] for i = 1:size(KJ_images, 3)]

err_y_hat = abs.(y_hat .- y_true)
err_y_hat_transpose = abs.(y_hat_transpose .- y_true)

println("L1 error y_hat: ", sum(err_y_hat))
println("L1 error y_hat transpose: ", sum(err_y_hat_transpose))