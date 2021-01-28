using Flux, Statistics
using Flux.Data: DataLoader
using Flux: throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDAapi
using BSON: @save
using Images, FileIO
using Plots
using TestImages

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
    epochs::Int = 120  # number of epochs
    device::Function = cpu  # set as gpu, if gpu available
end

function arr_to_input_output(arr)
    # Each column will be a element of our data set
    num_elements = size(arr, 1) * size(arr, 2)
    inputs = zeros(2, num_elements) # row, col as inputs 
    outputs = zeros(1, num_elements) # pixel value as output
    total_index = 1
    for i = 1:size(arr, 1)
        for j = 1:size(arr, 2)
            inputs[:, total_index] = [i; j]
            outputs[total_index] = arr[i, j] # access the appropriate matrix element
            total_index = total_index + 1
        end
    end
    return Float32.(inputs), Float32.(outputs)
end

function image_from_network(model, height, width, means, stdevs)
    arr = zeros(height, width)
    for i = 1:height
        for j = 1:width
            in_1 = (i - means[1]) / stdevs[1]
            in_2 = (j - means[2]) / stdevs[2]
            arr[i, j] = model([in_1, in_2])[1]
        end
    end
    return Gray.(arr)
end

function getdata(image_file, args)
    println("Loading data")
    img = Float64.(Gray.(testimage("mandrill")))
    inputs, outputs = arr_to_input_output(img)
    println("Means: ", mean(inputs, dims=2))
    println("stdev: ", std(inputs, dims=2))
    inputs = (inputs .- mean(inputs, dims=2)) ./ std(inputs, dims=2)
    println(size(inputs))
    println(size(outputs))
    # # Batching
    train_data = DataLoader(inputs, outputs, batchsize=args.batchsize, shuffle=true)
    # validation_data = DataLoader(validation_input, validation_labels, batchsize=args.batchsize, shuffle=false)
    # test_data = DataLoader(test_input, test_labels, batchsize=args.batchsize, shuffle=false)

    return train_data
end

function build_model(; layer_sizes, act)
    println("Building model")
    # ReLU except last layer softmax
    layers = Any[Dense(layer_sizes[i], layer_sizes[i+1], act) for i = 1:length(layer_sizes) - 2]
    push!(layers, Dense(layer_sizes[end-1], layer_sizes[end]))
    println("Layers: ", layers)
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

function eval_fcn!(m, train_data, train_loss, times, start_time, max_batch=Inf)
    # Compute the losses and accuracy
    new_train_loss = loss_all(train_data, m, max_batch)

    # Update your lists
    push!(train_loss, new_train_loss)
    push!(times, time() - start_time)

    # Print new things
    println("Train loss: ", new_train_loss)
end

function train(image_file, layer_sizes; output_label = "", act=relu, kws...)
    println("Starting train process")
    leaky_str = act==leakyrelu ? "_leaky" : ""
    # Initializing Model parameters
    args = Args(; kws...)

    # Load Data
    train_data = getdata(image_file, args)

    # Construct model
    m = build_model(layer_sizes=layer_sizes, act=act)
    train_data = args.device.(train_data)
    m = args.device(m)
    loss(x,y) = mse(m(x), y)

    ## Training
    # lists to store progress
    train_loss = []
    times = []
    start_time = time()

    # Setup the evaluation function
    evalcb = () -> eval_fcn!(m, train_data, train_loss, times, start_time, 200)

    # Choose your optimizer and train
    opt = ADAM(args.η)
    println("Starting training")
    ps = params(m)
    @epochs args.epochs Flux.train!(loss, ps, train_data, opt, cb = throttle(evalcb, 20))

    # Printout your evaluation metrics at each point in time
    println("Times: ", times)
    println("Train loss: ", train_loss)

    # Plot your accuracy and loss over time
    plot(times, train_loss, label="Train Loss", title=string(layer_sizes, " Loss over time"), xlabel="Time (s)", ylabel="CE Loss", legend=:topright)
    savefig(string("./ActivationRegionExperiments/Plots/", output_label, "_loss", layer_sizes, leaky_str, ".png"))

    @save "./ActivationRegionExperiments/Models/model_"*output_label*string("_", layer_sizes, leaky_str)*".bson" m

    # Example load syntax: BSON.load("./Models/model_-500_to_750_(5, 32, 64, 64, 32, 2).bson")[:m]
    # or @load "./Models/model_-500_to_750_(5, 32, 64, 64, 32, 2).bson" m
    return m
end

image_file = "./ActivationRegionExperiments/Images/IMG_5904.JPG"
layer_sizes = (2, 32, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 1024, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 64, 64, 32, 1)
model = train(image_file, layer_sizes; output_label="initial_test", act=relu)

img = Gray.(testimage("mandrill"))
means = [256.50214; 256.12494]
stdevs = [147.82558; 147.79298]
net_image = image_from_network(model, 512, 512, means, stdevs)
mosaicview(img, net_image)
