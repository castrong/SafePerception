# Functions from Kyle to interface with his neural net format
using BSON: @load

mutable struct NNet
    file::AbstractString
    weights::Array{Any,1}
    biases::Array{Any,1}
    symmetric::Float64
    numLayers::Int32
    inputSize::Int32
    outputSize::Int32
    maxLayerSize::Int32
    
    layerSizes::Array{Int32,1}
    mins::Array{Float64,1}
    maxes::Array{Float64,1}
    means::Array{Float64,1}
    ranges::Array{Float64,1}
    
    numNetworks::Int32
    numTau::Int32
    numPra::Int32
    tauCut::Array{Float64,1}
    praCut::Array{Float64,1}
    networkArray::Bool
end

# Added by CAS - assumes your BSON file has an attribute m with the model
function read_bson_network(file::AbstractString, input_mins, input_maxes, means, stdevs)
    # Read in BSON
    @load file m
    # Save to a nnet
    temp = tempname()
    flux2nnet(temp, m, input_mins, input_maxes, means, stdevs) # it calls for means and ranges but this should be ok? check this
    return read_network(temp)
end
    
function read_network(file::AbstractString)
    f = open(file)

    ###############
    # Begin reading .nnet file
    ###############

    #Read header line(s)
    line = readline(f)
    networkArray = false
    if (line[1] == 'A') || (line[4] =='A')
        networkArray = true
    end

    while line[1:2] == "//"
        line = readline(f)
    end

    numTau = -1
    numPra = -1
    tauCut = [-1.0]
    praCut = [-1.0]
    if networkArray
        #Read number of neural networks
        #line = readline(f)
        numNetworks = parse(Int32,record[1])

        #Read number of previous RA cutpoints
        line = readline(f)
        record = split(line,[',','\n'])
        numPra = parse(Int32,record[1])

        #Read each previous RA cutpoint
        line = readline(f)
        record = split(line,[',','\n'])
        praCut = zeros(numPra)
        for i=1:(numPra)
            praCut[i] = parse(Float64,record[i])
        end

        #Read the number of tau cutpoints
        line = readline(f)
        record = split(line,[',','\n'])
        numTau = parse(Int32,record[1])

        #Read each tau cutpoint
        line = readline(f)
        record = split(line,[',','\n'])
        tauCut = zeros(numTau)
        for i=1:(numTau)
            tauCut[i] = parse(Float64,record[i])
        end

    else
        numNetworks = 1
    end

    #Read information about the neural network
    #line = readline(f)
    record = split(line,[',','\n'])
    numLayers = parse(Int32,record[1])
    inputSize = parse(Int32,record[2])
    outputSize = parse(Int32,record[3])
    maxLayerSize=parse(Int32,record[4])

    line = readline(f)
    record = split(line,[',','\n'])
    layerSizes = zeros(Int32,numLayers+1)
    for i=1:(numLayers+1)
        layerSizes[i]=parse(Int32,record[i])
    end

    line = readline(f)
    # Outdated flag so ignore! 
    symmetric = 0.0
    # record = split(line,[',','\n'])
    # symmetric = parse(Float64,record[1])

    line = readline(f)
    record = split(line,[',','\n'])
    mins = zeros(inputSize)
    for i=1:(inputSize)
        mins[i]=parse(Float64,record[i])
    end

    line = readline(f)
    record = split(line,[',','\n'])
    maxes = zeros(inputSize)
    for i=1:(inputSize)
        maxes[i]=parse(Float64,record[i])
    end


    line = readline(f)
    record = split(line,[',','\n'])
    means = zeros(inputSize+1)
    for i=1:(inputSize+1)
        means[i]=parse(Float64,record[i])
    end

    line = readline(f)
    record = split(line,[',','\n'])
    ranges = zeros(inputSize+1)
    for i=1:(inputSize+1)
        ranges[i]=parse(Float64,record[i])
    end

    weights = Any[zeros(layerSizes[2],layerSizes[1])]
    biases  = Any[zeros(layerSizes[2])]
    for i=2:numLayers
        weights = [weights;Any[zeros(layerSizes[i+1],layerSizes[i])]]
        biases  = [biases;Any[zeros(layerSizes[i+1])]]
    end
    weights = Any[weights]
    biases  = Any[biases]
    for i=2:numNetworks
        w2 = Any[zeros(layerSizes[2],layerSizes[1])]
        b2 = Any[zeros(layerSizes[2])]
        for i=2:numLayers
            w2 = [w2;Any[zeros(layerSizes[i+1],layerSizes[i])]]
            b2 = [b2;Any[zeros(layerSizes[i+1])]]
        end
        weights = [weights;Any[w2]]
        biases  = [biases;Any[b2]]
    end

    nnetInd = 1
    layer=1
    i=1
    j=1
    line = readline(f)
    record = split(line,[',','\n'])
    while !eof(f)
        if layer>numLayers
            layer=1
            nnetInd=nnetInd+1
        end
        while i<=layerSizes[layer+1]
            while record[j]!=""
                weights[nnetInd][layer][i,j] = parse(Float64,record[j])
                j=j+1
            end
            j=1
            i=i+1
            line = readline(f)
            record = split(line,[',','\n'])
        end
        i=1
        while i<=layerSizes[layer+1]
            biases[nnetInd][layer][i] = parse(Float64,record[1])
            i=i+1
            line = readline(f)
            record = split(line,[',','\n'])
        end
        layer=layer+1
        i=1
        j=1
    end
    close(f)

    return NNet(file,weights,biases,symmetric,numLayers,inputSize,outputSize,maxLayerSize,layerSizes,mins,maxes,means,ranges,numNetworks,numTau,numPra,tauCut,praCut,networkArray)
end

#Evaluates one set of inputs
function evaluate_network(nnet::NNet,input::Array{Float64,1})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    symmetric = nnet.symmetric
    biases = nnet.biases
    weights = nnet.weights
    
    nnetInd = 1
    if nnet.numNetworks>1
        tau = input[6]
        pra = input[7]
        nnetInd += round(Int32,pra) -1
        nnetInd *= nnet.numTau
        
        upper = nnet.numTau
        lower = 1
        while lower < upper
            middle = floor(Int32,(upper+lower)/2)
            if tau - nnet.tauCut[middle] > nnet.tauCut[middle+1]-tau
                lower = middle + 1
            else
                upper = middle
            end
        end
        nnetInd += lower
    end
    
    inputs = zeros(inputSize)
    for i = 1:inputSize
        if input[i]<nnet.mins[i]
            inputs[i] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
        elseif input[i] > nnet.maxes[i]
            inputs[i] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
        else
            inputs[i] = (input[i]-nnet.means[i])/nnet.ranges[i] 
        end
    end

    for layer = 1:numLayers-1
        temp = max.(weights[nnetInd][layer]*inputs[1:nnet.layerSizes[layer]]+biases[nnetInd][layer],0)
        inputs = temp
    end
    outputs = weights[nnetInd][end]*inputs[1:nnet.layerSizes[end-1]]+biases[nnetInd][end]
    for i=1:outputSize
        outputs[i] = outputs[i]*nnet.ranges[end]+nnet.means[end]
    end
    return outputs
end

#Evaluates multiple inputs at once. Each set of inputs should be a column in the input array
#Returns a column of output Q values for each input set
function evaluate_network_multiple(nnet::NNet,input::Array{Float64,2})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    symmetric = nnet.symmetric
    biases = nnet.biases
    weights = nnet.weights
        
    _,numInputs = size(input)
    symmetryVec = zeros(numInputs)
    
    
    nnetInd = 1
    if nnet.numNetworks>1
        tau = input[6]
        pra = input[7]
        nnetInd += round(Int32,pra) - 1
        nnetInd *= nnet.numTau
        
        upper = nnet.numTau
        lower = 1
        while lower < upper
            middle = floor(Int32,(upper+lower)/2)
            if tau - nnet.tauCut[middle] > nnet.tauCut[middle+1]-tau
                lower = middle + 1
            else
                upper = middle
            end
        end
        nnetInd += lower
    end
    
    inputs = zeros(inputSize,numInputs)
    for i = 1:inputSize
        for j = 1:numInputs
            if input[i,j]<nnet.mins[i]
                inputs[i,j] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
            elseif input[i,j] > nnet.maxes[i]
                inputs[i,j] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
            else
                inputs[i,j] = (input[i,j]-nnet.means[i])/nnet.ranges[i] 
            end
        end
    end

    for layer = 1:numLayers-1
        inputs = max.(weights[nnetInd][layer]*inputs[1:nnet.layerSizes[layer],:]+biases[nnetInd][layer]*ones(1,numInputs),0)
    end
    outputs = weights[nnetInd][end]*inputs[1:nnet.layerSizes[end-1],:]+biases[nnetInd][end]*ones(1,numInputs)
    for i=1:outputSize
        for j=1:numInputs
            outputs[i,j] = outputs[i,j]*nnet.ranges[end]+nnet.means[end]
        end
    end
    return outputs
end

# NOTE: ASSUMES INPUTS ARE NORMALIZED!!!!!!!
function evaluate_network_faster(nnet::NNet,input::Array{Float64,2})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    symmetric = nnet.symmetric
    biases = nnet.biases
    weights = nnet.weights
        
    _,numInputs = size(input)
    symmetryVec = zeros(numInputs)
    
    
    nnetInd = 1
    
    inputs = input

    for layer = 1:numLayers-1
        inputs = max.(weights[nnetInd][layer]*inputs[1:nnet.layerSizes[layer],:]+biases[nnetInd][layer]*ones(1,numInputs),0)
    end

    outputs = weights[nnetInd][end]*inputs[1:nnet.layerSizes[end-1],:]+biases[nnetInd][end]*ones(1,numInputs)

    for i=1:outputSize
        for j=1:numInputs
            outputs[i,j] = outputs[i,j]*nnet.ranges[end]+nnet.means[end]
        end
    end

    return outputs
end

function num_inputs(nnet::NNet)
    if nnet.numNetworks==1
        return nnet.inputSize
    else
        return nnet.inputSize+2
    end
end
function num_outputs(nnet::NNet)
    return nnet.outputSize
end

# Added by CAS
# NOTE: ASSUMES INPUTS ARE NORMALIZED!!!!!!!
(nnet::NNet)(input) = return evaluate_network_faster(nnet, reshape(input, (length(input), 1)))

"""
Functions added by SMK
"""

function convert_nn_to_advs(nτ)
    τs = collect(0:1:nτ-1)
    grid = RectangleGrid(hs, ḣ₀s, ḣ₁s, actions, τs)
    advs = BitArray(undef, length(grid), length(actions))

    nnets = Dict()
    # Iterate over pras to load networks
    for pra = actions
        nnets[pra] = read_network("networks_dasc/VertCAS_newTransIntrSpeed_pra0$(pra+1)_v5_25HU_1000.nnet")
    end

    # Printing
    fracBase = 0.05
    frac = fracBase

    n = size(advs, 1)
    # Iterate over states to get advisories
    for i = 1:n
        # Printing
        if i/n ≥ frac
            print(round(frac*100))
            println("% Complete")
            frac += fracBase
        end

        s = ind2x(grid, i)
        h, ḣ₀, ḣ₁, pra, τ = s[1], s[2], s[3], s[4], s[5]
        input = [h, ḣ₀, ḣ₁, τ]
        q = evaluate_network(nnets[pra], input)
        advs[i, argmax(q)] = true
    end

    return advs
end