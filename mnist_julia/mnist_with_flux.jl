# mnist with flux
# I choose the architecture given in this article: 
# https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
# at the end of the file, I ploted the evolution loss through epoch

using Flux, MLDatasets, ProgressMeter, Plots

function encode_in_one_hot(x)
    array = zeros(10)
    array[x + 1] = 1 # because array index start at 1 in julia (lol)
    return array
end

show(encode_in_one_hot(5))

function get_data()
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]

    oneHotYTrain = Array{Float32, 2}(undef, 10, 60000)
    oneHotYTest = Array{Float32, 2}(undef, 10, 10000)

    for i in 1:last(size(ytrain))
        oneHotYTrain[:, i] = encode_in_one_hot(ytrain[i])
        if i <= 10000
            oneHotYTest[:, i] = encode_in_one_hot(ytest[i])
        end
    end

    (
        (reshape(xtrain, 28, 28, 1, :), reshape(oneHotYTrain, 10, :)),
        (reshape(xtest, 28, 28, 1, :), reshape(oneHotYTest, 10, :)),
    )

end

(xtrain, ytrain), (xtest, ytest) = get_data();

model = Chain(Conv((3, 3), 1 => 32, Flux.relu),
            Conv((3, 3), 32 => 64, Flux.relu),
            MaxPool((2,2)),
            Dropout(0.25),
            Flux.flatten,
            Dense(9216 => 128, relu),
            Dropout(0.5),
            Dense(128 => 10, relu),
            softmax)

opt_state = Flux.setup(Adam(), model)

# split it with batchsize
traindata = Flux.Data.DataLoader((data=xtrain, label=ytrain), batchsize=32, shuffle=true) 

losses = []
@showprogress for epoch in 1:10
    for (x, y) in traindata
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

x = 1:last(size(losses))
show(losses)
plot(x, losses)

savefig("mnist_lossplot.png")


