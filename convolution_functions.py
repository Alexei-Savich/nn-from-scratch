import pickle

import numpy as np
import pandas as pd
import sklearn

import NN as nn
from autoencoder_main import convert_to_image
import matplotlib.pyplot as plt


def sigmoid(X, derivative=False):
    if derivative:
        sig = 1 / (1 + np.exp(-X))
        return sig * (1 - sig)
    else:
        return 1 / (1 + np.exp(-X))


def relu(X, derivative=False):
    if derivative:
        return np.where(X > 0, 1, 0)
    else:
        return np.maximum(0, X)


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    b = np.squeeze(b)
    Z = Z + b
    return Z


def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vert_start = stride * h
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    cache = (A_prev, W, b, hparameters)
    return Z, cache


def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        if not pad == 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db


def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask


def distribute_value(dz, shape):
    average = np.prod(shape)
    a = (dz / average) * np.ones(shape)
    return a


def pool_backward(dA, cache, mode="max"):
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        a_prev = A_prev[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    assert (dA_prev.shape == A_prev.shape)
    return dA_prev


def pool_forward(A_prev, hparameters, mode="max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        a_prev_slice = A_prev[i]
        for h in range(n_H):
            vert_start = stride * h
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_slice_prev = a_prev_slice[vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)
                    else:
                        raise RuntimeError('Incorrect pooling mode: ' + mode)
    cache = (A_prev, hparameters)
    assert (A.shape == (m, n_H, n_W, n_C))
    return A, cache


def update_parameters(parameters, gradients, learning_rate):
    updated_parameters = {}
    for key in parameters.keys():
        updated_parameters[key] = parameters[key] - learning_rate * gradients[key]
    return updated_parameters


def train(X_train, network: nn.NeuralNetwork, parameters, number_of_epochs, mini_batch_size, hparameters_conv1,
          hparameters_pool, hparameters_conv2, learning_rate):
    m = X_train.shape[0]
    for epoch in range(number_of_epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X_train[permutation]
        for i in range(0, m, mini_batch_size):
            X_mini = X_shuffled[i:i + mini_batch_size]

            A, cache_conv1 = conv_forward(X_mini, parameters["W1"], parameters["b1"], hparameters_conv1)
            A = relu(A)
            P, cache_pool = pool_forward(A, hparameters_pool, mode="max")
            Z, cache_conv2 = conv_forward(P, parameters["W2"], parameters["b2"], hparameters_conv2)
            Z = sigmoid(Z)

            n = Z.shape[0]
            Z = Z.reshape(n, -1)

            network.feed_forward(Z)
            dZ = network.back_propagation(Z, X_mini.reshape(X_mini.shape[0], -1), 0.000009)  # -------------------------------------------------------------------------------------------------------

            dZ = dZ * sigmoid(Z, derivative=True)
            dZ = dZ.reshape(dZ.shape[0], 4, 4, 5)

            dP, dW2, db2 = conv_backward(dZ, cache_conv2)
            dA = pool_backward(dP, cache_pool, mode="max")
            dA = dA * relu(A, derivative=True)
            _, dW1, db1 = conv_backward(dA, cache_conv1)

            gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
            parameters = update_parameters(parameters, gradients, learning_rate)

        loss = network.calc_loss(network.feed_forward(Z), X_mini.reshape(X_mini.shape[0], -1))
        print(f"Epoch {epoch + 1}/{number_of_epochs}, Loss: {loss}")

        with open('CNN.pkl', 'wb') as file:
            pickle.dump(parameters, file)
        with open('FCN.pkl', 'wb') as file:
            pickle.dump(network, file)


def plot_cnn_filters():
    with open('CNN_4.pkl', 'rb') as file:
        parameters = pickle.load(file)
    W1 = parameters['W1']
    num_filters = W1.shape[-1]
    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
    for i in range(num_filters):
        filter = W1[:, :, 0, i]
        ax = axes[i]
        ax.imshow(filter, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i + 1}')
    plt.show()


def main():
    weights_strategy = 'xavier'

    input_size1 = 28 * 28
    weight_size1 = 27
    weight_size2 = 135
    bias_size1 = 3
    bias_size2 = 5
    input_size2 = 6 * 6

    if weights_strategy == 'xavier':
        lower, upper = -(1.0 / np.sqrt(input_size1)), (1.0 / np.sqrt(input_size1))
        weights1 = lower + np.random.rand(weight_size1) * (upper - lower)
        bias1 = lower + np.random.rand(bias_size1) * (upper - lower)

        lower, upper = -(1.0 / np.sqrt(input_size2)), (1.0 / np.sqrt(input_size2))
        weights2 = lower + np.random.rand(weight_size2) * (upper - lower)
        bias2 = lower + np.random.rand(bias_size2) * (upper - lower)
    elif weights_strategy == 'he':
        std = np.sqrt(2.0 / input_size1)
        weights1 = np.random.randn(weight_size1) * std
        bias1 = np.random.randn(bias_size1) * std

        std = np.sqrt(2.0 / input_size2)
        weights2 = np.random.randn(weight_size2) * std
        bias2 = np.random.randn(bias_size2) * std
    elif weights_strategy == 'load':
        with open('CNN.pkl', 'rb') as f:
            parameters = pickle.load(f)
        with open('FCN.pkl', 'rb') as f:
            network = pickle.load(f)

    if not weights_strategy == 'load':
        parameters = {
            "W1": weights1.reshape(3, 3, 1, 3),
            "b1": bias1.reshape(1, 1, 1, 3),
            "W2": weights2.reshape(3, 3, 3, 5),
            "b2": bias2.reshape(1, 1, 1, 5)
        }
        network = nn.NeuralNetwork([80, 180, 784], [None, 'sigmoid', 'linear'],
                                   weights_strategy='xavier')

    learning_rate = 0.000083  # ---------------------------------------------------------------------------------------------------------------
    number_of_epochs = 1
    mini_batch_size = 64

    hparameters_conv1 = {"stride": 2, "pad": 0}
    hparameters_pool = {"stride": 2, "f": 2}
    hparameters_conv2 = {"stride": 1, "pad": 0}

    data1 = pd.read_csv('fashion-mnist_train.csv')
    data2 = pd.read_csv('fashion-mnist_test.csv')
    data = pd.concat([data1, data2])
    X = data[data['label'] == 8]
    y = X[['label']].values
    X = X.drop(columns=['label']).values / 255

    X = X.reshape(X.shape[0], 28, 28, 1)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=0.80)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val, y_val, train_size=0.5)
    train(X_train, network, parameters, number_of_epochs, mini_batch_size, hparameters_conv1, hparameters_pool,
          hparameters_conv2, learning_rate)


def f_mnist(value=4):
    with open(f'CNN_{value}.pkl', 'rb') as f:
        parameters = pickle.load(f)
    with open(f'FCN_{value}.pkl', 'rb') as f:
        network = pickle.load(f)

    hparameters_conv1 = {"stride": 2, "pad": 0}
    hparameters_pool = {"stride": 2, "f": 2}
    hparameters_conv2 = {"stride": 1, "pad": 0}

    data1 = pd.read_csv('fashion-mnist_train.csv')
    data2 = pd.read_csv('fashion-mnist_test.csv')
    data = pd.concat([data1, data2])
    X = data[data['label'] == value]
    y = X[['label']].values
    X = X.drop(columns=['label']).values / 255

    X = X.reshape(X.shape[0], 28, 28, 1)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=0.85)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val, y_val, train_size=0.8)

    A, cache_conv1 = conv_forward(X_test, parameters["W1"], parameters["b1"], hparameters_conv1)
    A = relu(A)
    P, cache_pool = pool_forward(A, hparameters_pool, mode="max")
    Z, cache_conv2 = conv_forward(P, parameters["W2"], parameters["b2"], hparameters_conv2)
    Z = sigmoid(Z)

    n = Z.shape[0]
    Z = Z.reshape(n, -1)

    network.feed_forward(Z)

    y_pred = network.feed_forward(Z)

    print("Loss of the best model:")
    print(network.calc_loss(y_pred, X_test.reshape(X_test.shape[0], -1)).mean())

    convert_to_image(y_pred)


if __name__ == "__main__":
    # main()
    # f_mnist()
    plot_cnn_filters()