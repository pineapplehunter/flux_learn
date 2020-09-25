module mnist

using Flux, Flux.Data.MNIST
using Flux: onehotbatch, onecold, logitcrossentropy, @epochs
using Statistics
using BSON: @save

function load_mnist_data()
    function images2array(imgs; shape = (28, 28, 1))
        N = size(imgs)[1]
        ret = Array{Float32}(undef, shape..., size(imgs)[1])
        for i = 1:N
            ret[:, :, :, i] = Float32.(imgs[i])
        end
        ret
    end

    X = images2array(MNIST.images())
    Xtest = images2array(MNIST.images(:test))

    function label2onehot(labels, to = 0:9)
        onehotbatch(labels, to)
    end

    Y = label2onehot(MNIST.labels())
    Ytest = label2onehot(MNIST.labels(:test))
    X, Xtest, Y, Ytest
end

function make_model()
    Chain(
        Conv((3, 3), 1 => 8, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 8, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(200, 64, relu),
        Dense(64, 10),
        logsoftmax,
    )
end

function accuracy(x, y, m)
    mean(onecold(m(x)) .== onecold(y))
end

function train()
    X, Xtest, Y, Ytest = load_mnist_data()
    data = Flux.Data.DataLoader(X, Y, batchsize = 32)

    model = make_model()
    loss(x, y) = logitcrossentropy(model(x), y)
    opt = Flux.Optimise.ADAM()

    function status()
        @info("loss \t= $(loss(Xtest,Ytest))")
        @info("accuracy \t= $(accuracy(Xtest,Ytest,model))")
    end

    @epochs 1 begin
        Flux.train!(loss, params(model), data, opt)
        status()
    end

    @save "model.bson" model

    return model
end

end