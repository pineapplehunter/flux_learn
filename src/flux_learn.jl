module flux_learn

using Flux
using Plots

greet() = print("Hello World!")

function create_model(; input=1, output=1)
    h = 64
    activation = relu
    Chain(Dense(input, h, activation;initb=randn), Dense(h, h, activation), Dense(h, output))
end

function train(model)
    r = collect(-1:0.001:1)
    X = reshape(r, 1, length(r))
    Y = sin.(X .* 10)

    data = Flux.Data.DataLoader(X, Y, batchsize=32, shuffle=true)

    loss(x, y) = Flux.Losses.mse(model(x), y)
    opt = ADAM(0.01)

    # precompile model
    model(rand(1, 1))
    p = params(model)

    @gif for i ∈ 1:20
        @info "$i epochs"
        Flux.train!(loss, p, data, opt)
        plot(r, Y')
        plot!(r, model(X)')
        plot!(title="frame $i", xlim=(-1, 1), ylim=(-1.2, 1.2))
    end every 1
end

function main()
    model = create_model(; output=1)

    range = collect(-1.5:0.01:1.5)
    X = reshape(range, 1, length(range))

    model = train(model)

    output = model(X)
    @show size(output)
    # @show params(model)
    plot(range, output', label="predicted")
    plot!(range, sin.(range .* 10), label="sin(10x)")
    plot!(title="model prediction")
    # output = reshape(output, length(output))
    # plot(range, output)
    # output
end
end
