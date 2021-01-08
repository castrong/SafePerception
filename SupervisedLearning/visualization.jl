### Visualize what a model has learned ###
using Flux
using BSON
using Colors
using Reel
using HDF5
using Plots
Reel.extension(m::MIME"image/svg+xml") = "svg"
Reel.set_output_type("gif") # may be necessary for use in IJulia


### Get data and load the model ###
# Load the models
mlinear = BSON.load("./Models/linear_3-128.bson")[:m]
mse_net = BSON.load("./Models/relu_3-32-32-64-128.bson")[:m]
mae_net = BSON.load("./Models/relu_mae_3-32-32-64-128.bson")[:m]
mpe_5layer = BSON.load("./Models/max_pixel_error_3-32-32-64-128.bson")[:m]

mae_big = BSON.load("./Models/relu_mae_big-3-32-32-64-64-64-64-128-128-128.bson")[:m]
mae_22layer = BSON.load("./Models/relu_mae_22layer-3-32-32-64-64-64-64-64-64-64-64-64-64-64-64-64-64-64-64-128-128-128.bson")[:m]
model = mlinear
# models = [mlinear, mse_net, mae_net, mpe_5layer, mae_big, mae_22layer]
# model_names = ["Linear", "MSE_3-32-32-64-128", "MPE_3-32-32-64-128", "MAE_3-32-32-64-128", "MAE_3-32-32-64-64-64-64-128-128-128", "MAE_22layer-3-32-32-64-64-64-64-64-64-64-64-64-64-64-64-64-64-64-64-128-128-128"]
models = [mpe_5layer]
model_names = ["MPE_3-32-32-64-128"]

# Loading Dataset
train_filename = "./Data/SK_DownsampledTrainingData.h5"
val_filename = "./Data/SK_DownsampledValidationData.h5"
x_train = h5read(train_filename, "X_train")
y_train = h5read(train_filename, "y_train")
x_val = h5read(val_filename, "X_val")
y_val = h5read(val_filename, "y_val")

# Normalize the features of the input
x_train[1, :] /= 20.0 # Crosstrack error
x_val[1, :] /= 20.0
x_train[2, :] /= 80.0 # Heading error
x_val[2, :] /= 80.0
x_train[3, :] = x_train[3, :] ./ 2960 .- 0.4213     # Position down the runway
x_val[3, :] = x_val[3, :] ./ 2960 .- 0.4213

### Compare some inputs ###
x1 = x_val[:, 1]
y_hat = reshape(model(x1), (16, 8))
y = y_val[:, :, 1]
Gray.(y_hat)
Gray.(y)
Gray.(abs.(y_hat - y))

### Create an animation moving down the runway ###
start_pos = -0.1
end_pos = 0.1
steps = 100
positions = LinRange(start_pos, end_pos, steps)

### Create an animation walking through the validation set ###
steps = 100
for (i, model) in enumerate(models)
    frames = Frames(MIME("image/svg+xml"), fps=10)
    for i = 1:steps
        cur_x = x_val[:, i*5]
        cur_y_hat = reshape(model(cur_x), (16, 8))
        # push!(frames_hat, Gray.(cur_y_hat'))
        # push!(frames_groundtruth, Gray.(y_val[:, :, i]'))
        color = :greys
        cur_plot = plot(heatmap(reverse(cur_y_hat', dims=1), xlabel="Predicted", clims=(0, 1), color=color, aspect_ratio=:equal, xaxis=false, yaxis=false, legend=:none),
                        heatmap(reverse(y_val[:, :, i*5]', dims=1), xlabel="Ground Truth", clims=(0, 1), color=color, aspect_ratio=:equal, xaxis=false, yaxis=false, legend=:none))
        push!(frames, cur_plot)
    end
    save_name = "./Visualizations/val_path_predict_"*model_names[i]*".gif"
    println("Writing gif for model ", save_name)
    write(save_name, frames)
end
