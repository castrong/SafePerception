using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, throttle, @epochs, mse, mae
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDAapi
using MLDatasets
using CSV
using DataFrames
using Plots
using BSON: @save
using HDF5
using PGFPlots

# Built off of Flux Model zoo MLP for MNIST
# https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    import CuArrays		# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw mutable struct Args
    η::Float64 = 0.001  # 3e-4 was here    # learning rate
    batchsize::Int = 1024  # batch size
    epochs::Int = 10    # number of epochs
end

function getdata(args)
    println("Loading data")
    train_filename = "./Data/SK_DownsampledTrainingData.h5"
    val_filename = "./Data/SK_DownsampledValidationData.h5"

    # Loading Dataset - the batch loader will
    # assume each column is an observation so we'll need
    # to make sure that's true
    x_train = h5read(train_filename, "X_train")
    y_train = h5read(train_filename, "y_train")
    x_val = h5read(val_filename, "X_val")
    y_val = h5read(val_filename, "y_val")

    # Reshape the outputs into im_height*im_width x num_observations
    y_train = reshape(y_train, (16*8, :))
    y_val = reshape(y_val, (16*8, :))

    # Normalize the features of the input
    x_train[1, :] /= 20.0 # Crosstrack error
    x_val[1, :] /= 20.0
    x_train[2, :] /= 80.0 # Heading error
    x_val[2, :] /= 80.0
    x_train[3, :] = x_train[3, :] ./ 2960 .- 0.4213     # Position down the runway
    x_val[3, :] = x_val[3, :] ./ 2960 .- 0.4213

    println("Max after: ", maximum(x_train, dims=2))
    println("Min after: ", minimum(x_train, dims=2))
    print("size x: ", size(x_train), size(x_val))
    print("size y: ", size(y_train), size(y_val))

    # Batching
    train_data = DataLoader(x_train, y_train, batchsize=args.batchsize, shuffle=true)
    validation_data = DataLoader(x_val, y_val, batchsize=args.batchsize, shuffle=true)

    return train_data, validation_data
end

function build_model(; layer_sizes, act)
    # ReLU except last layer identity
    layers = Any[Dense(layer_sizes[i], layer_sizes[i+1], act) for i = 1:length(layer_sizes) - 2]
    push!(layers, Dense(layer_sizes[end-1], layer_sizes[end]))
    println("Model created with layers: ", layers)
    return Chain(layers...)
end

function loss_all(dataloader, model, max_batch=Inf)
    l = 0f0
    i = 0
    for (x,y) in dataloader
        l += mse(model(x), y)
        i = i + 1
        i >= max_batch && break
    end
    l/i
end

# Fcn called during training every __ seconds
function eval_fcn!(m, train_data, validation_data, train_loss, val_loss, times, start_time, max_batch=Inf)
    # Compute the losses and accuracy
    new_train_loss = loss_all(train_data, m, max_batch)
    new_val_loss = loss_all(validation_data, m, max_batch)

    # Update your lists
    push!(train_loss, new_train_loss)
    push!(val_loss, new_val_loss)
    push!(times, time() - start_time)

    # Print updated losses
    println("Train loss: ", new_train_loss, " Validation loss: ", new_val_loss)
end

function train(layer_sizes, l_in, save_file; kws...)
    println("Starting train process")
    # Initializing Model parameters
    args = Args(; kws...)

    # Load Data
    train_data, validation_data = getdata(args)

    # Construct model and loss
    m = build_model(layer_sizes=layer_sizes, act=relu)
    loss(x,y) = l_in(m(x), y)

    ## Training
    # lists to store progress
    train_loss = []
    val_loss = []
    times = []
    start_time = time()

    # Setup the evaluation function
    evalcb = () -> eval_fcn!(m, train_data, validation_data, train_loss, val_loss, times, start_time, 100)

    # Choose your optimizer and train
    opt = ADAM(args.η)
    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = throttle(evalcb, 20))

    # Save the model
    @save save_file m

    # Printout your evaluation metrics at each point in time
    println("Times: ", times)
    println("Train loss: ", train_loss)
    println("Val loss: ", val_loss)

    # Show the final loss on train and validation data
    @show loss_all(train_data, m)
    @show loss_all(validation_data, m)

    # Plot your loss over time
    Plots.plot(times, train_loss, label="Train Loss", title=string(layer_sizes, " Loss over time"), xlabel="Time (s)", ylabel="CE Loss", legend=:topright, yaxis=:log)
    Plots.plot!(times, val_loss, label="Validation Loss")
    savefig(string("loss", layer_sizes, ".png"))
end

max_pixel_error(y_hat, y, agg=mean) = agg(maximum(y_hat .- y, dims=1))

cd(@__DIR__)
layer_sizes = (3, 32, 32, 64, 128)
save_file = "./Models/max_pixel_error_"*string(layer_sizes)*".bson"
loss = max_pixel_error
train(layer_sizes, mae, save_file)
