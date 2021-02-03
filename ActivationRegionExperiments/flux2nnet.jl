function flux2nnet(filename, model, inputMins, inputMaxes, inputMeans, inputRanges, outputMean=0.0, outputRange=1.0)
    num_layers = length(model) + 1
    weights = Flux.params(model)

    layer_sizes = zeros(Int, num_layers)
    layer_sizes[1] = size(weights[1], 2)
    for i = 1:num_layers - 1
        layer_sizes[i + 1] = length(weights[2 * i])
    end

    open(filename, "w") do f
        write(f, "// Converted from Flux.jl format\n")

        write(f, "$(num_layers - 1),")
        write(f, "$(layer_sizes[1]),")
        write(f, "$(layer_sizes[end]),")
        write(f, "$(maximum(layer_sizes)),\n")

        for i = 1:num_layers
            write(f, "$(layer_sizes[i]),")
        end
        write(f, "\n")

        write(f, "0,\n")

        for i = 1:length(inputMins)
            write(f, "$(inputMins[i]),")
        end
        write(f, "\n")

        for i = 1:length(inputMaxes)
            write(f, "$(inputMaxes[i]),")
        end
        write(f, "\n")

        for i = 1:length(inputMeans)
            write(f, "$(inputMeans[i]),")
        end
        write(f, "$(outputMean),")
        write(f, "\n")

        for i = 1:length(inputRanges)
            write(f, "$(inputRanges[i]),")
        end
        write(f, "$(outputRange),")
        write(f, "\n")

        for weight in weights
            write_matrix(f, weight)
        end
    end
end

function write_matrix(f, mat)
    for i = 1:size(mat, 1)
        for j = 1:size(mat, 2)
            write(f, "$(mat[i, j]),")
        end
        write(f, "\n")
    end
end