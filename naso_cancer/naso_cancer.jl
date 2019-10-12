include("/home/tongxueqing/zhaox/MachineLearning/Julia_ML/densenets.jl")
using .Densenet
using Random
using Statistics
using PyCall
using Flux
using BSON: @save
sio = PyCall.pyimport("scipy.io")

function load_data(series; path, ratio = 0.9, batch = 64)
    filenames = [filename for filename in readdir(path) if occursin(series * ".mat", filename)]
    len = length(filenames)
    shuffleOrder = shuffle(1:len)
    trainnames = filenames[shuffleOrder[1:Int(floor(ratio * len))]]
    testnames = filenames[shuffleOrder[Int(ceil(ratio * len)):len]]
    testnames = [filename for filename in testnames if occursin("0.rotate", filename)]
    trainbatches = []
    for i in 1:Int(floor(length(trainnames) // batch) + 1)
        push!(trainbatches, trainnames[(i - 1) * batch + 1:min(i * batch, length(trainnames))])
    end
    testbathes = []
    for i in 1:Int(floor(length(testnames) // batch) + 1)
        push!(testbathes, testnames[(i - 1) * batch + 1: min(i * batch, length(testnames))])
    end
    trainbatches, testbathes
end

function label_smooth(y, K; smooth = 0.01, weight = Nothing)
    lsy = (1 - smooth) * Flux.onehotbatch(y .+ 1, 1:K) .+ smooth / K
    if weight === Nothing
        weight = ones(K, 1)
    end
    lsy .* weight
end

function accuracy(batches, path, net)
    correct = [0, 0]
    for (j, filebatch) in enumerate(batches)
        X = cat([sio.loadmat(path * file)["data"] for file in filebatch]..., dims = 4)
        X = permutedims(X, [2, 3, 1, 4]) |> gpu
        Y = ifelse.(startswith.(filebatch, "1"), 1, 0) |> gpu
        correct[1] += sum(Flux.onecold(net(X)) .== (Y .+ 1))
        correct[2] += length(Y)
    end
    println("accuracy in train: $(correct[1] / correct[2])")
end

function main(series, num_iterations; batch = 64, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999)
    path = "/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/cut_slice/"
    trainbatches, testbatches = load_data(series, path = path, batch = batch)
    net = Densenet.get_densenet_model(121)
    weight = [0.84, 0.16]
    loss(x, y) = Flux.crossentropy(net(x), y)
    parameters = params(net)
    optimizer = AdaMax(learning_rate, (beta1, beta2))
    data = []
    for (j, filebatch) in enumerate(trainbatches)
        X = cat([sio.loadmat(path * file)["data"] for file in filebatch]..., dims = 4)
        X = permutedims(X, [2, 3, 1, 4])
        Y = label_smooth(ifelse.(startswith.(filebatch, "1"), 1, 0), 2, weight = weight)
        push!(data, (X, Y))
    end
    for i in 1:num_iterations
        if i % 30 == 0
            learning_rate /= 10
            optimizer = AdaMax(learning_rate, (beta1, beta2))
        end
        Flux.train!(loss, parameters, (d |> gpu for d in data), optimizer, cb = () -> accuracy(trainbatches, path, net))
    end
    @save "/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/julia.bson" net
    accuracy(trainbatches, path, net)
    accuracy(testbatches, path, net)
end

main("1", 90, batch = 32)