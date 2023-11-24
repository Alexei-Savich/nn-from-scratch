import os
import pickle

import numpy as np
import pandas as pd
import sklearn.model_selection
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score
from scipy.ndimage import rotate

import NN as nn


def convert_to_image(array, folder='results'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    images = []
    for i, row in enumerate(array):
        image = row.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        file_path = os.path.join(folder, f'image_{i}.png')
        plt.savefig(file_path)
        plt.close()
        images.append(file_path)
    return images


def visualize():
    label_names = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    with open('autoencoder_loss_0_0159', 'rb') as f:
        network = pickle.load(f)
    data1 = pd.read_csv('fashion-mnist_train.csv')
    data2 = pd.read_csv('fashion-mnist_test.csv')
    data = pd.concat([data1, data2])
    data = data[(data['label'] == 4) | (data['label'] == 5) | (data['label'] == 1) | (data['label'] == 8)]
    labels = data['label'].map(label_names).values
    X = data.drop(columns=['label']).values / 255
    #
    X = X.reshape(-1, 28, 28)
    augmented_X = []
    for image in X:
        angle = np.random.uniform(-30, 30)
        rotated_image = rotate(image, angle, reshape=False)
        augmented_X.append(rotated_image)
    augmented_X = np.array(augmented_X)
    X = augmented_X.reshape(-1, 784)

    # X += np.random.normal(0, 0.05, X.shape)
    # after_noise = network.feed_forward(X)

    # network.visualize_hidden_layers(X)

    network.visualize_umap(X, labels, 'X')
    hidden_layer_features = network.get_hidden_layer_outputs(X)
    for i, layer_features in enumerate(hidden_layer_features):
        network.visualize_umap(layer_features, labels, i)

    convert_to_image(X)


def train_with_encoder():
    with open('autoencoder_loss_0_0159', 'rb') as f:
        network = pickle.load(f)
    data1 = pd.read_csv('fashion-mnist_train.csv')
    data2 = pd.read_csv('fashion-mnist_test.csv')
    data = pd.concat([data1, data2])

    encoded = []
    y = []

    for i in range(10):
        X = data[data['label'] == i].drop(columns=['label']).values / 255
        hidden_layer_features = network.get_hidden_layer_outputs(X)
        encoded.append(hidden_layer_features[1])
        y.extend([i] * len(hidden_layer_features[1]))

    encoded = np.vstack(encoded)
    y = np.array(y)

    num_classes = 10
    one_hot_encoded = np.zeros((len(y), num_classes))
    for i, item in enumerate(y):
        one_hot_encoded[i, item] = 1
    y = one_hot_encoded

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(encoded, y, train_size=0.85)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val, y_val, train_size=0.8)

    network = nn.NeuralNetwork([32, 100, 100, 100, 10], [None, 'sigmoid', 'relu', 'sigmoid', 'softmax'],
                               weights_strategy='he')
    network = network.train(X_train, y_train, X_val, y_val, learning_rate=0.0005, epochs=200)

    y_pred = network.feed_forward(X_test)

    print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    print("Loss of the best model:")
    print(np.average(np.sum(network.calc_loss(network.feed_forward(X_test), y_test), axis=1)))
    print("Precision:")
    print(precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro'))
    print("F1:")
    print(f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro'))
    print("Accuracy:")
    print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))


def train_no_encoder():
    network = nn.NeuralNetwork([784, 100, 100, 100, 10], [None, 'sigmoid', 'relu', 'sigmoid', 'softmax'],
                               weights_strategy='he')

    data1 = pd.read_csv('fashion-mnist_train.csv')
    data2 = pd.read_csv('fashion-mnist_test.csv')
    data = pd.concat([data1, data2])

    X = data.drop(columns=['label']).values
    y = data[['label']].values

    num_classes = 10
    one_hot_encoded = np.zeros((len(y), num_classes))
    for i, item in enumerate(y):
        one_hot_encoded[i, item[0]] = 1
    y = one_hot_encoded

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=0.8)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val, y_val, train_size=0.5)

    network = network.train(X_train, y_train, X_val, y_val, learning_rate=0.0005, epochs=200, early_stop_epochs=4)

    y_pred = network.feed_forward(X_test)

    print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    print("Loss of the best model:")
    print(np.average(np.sum(network.calc_loss(network.feed_forward(X_test), y_test), axis=1)))
    print("Precision:")
    print(precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro'))
    print("F1:")
    print(f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro'))
    print("Accuracy:")
    print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))


def f_mnist():
    network = nn.NeuralNetwork([784, 128, 32, 128, 784], [None, 'sigmoid', 'sigmoid', 'sigmoid', 'linear'],
                               weights_strategy='xavier')
    data1 = pd.read_csv('fashion-mnist_train.csv')
    data2 = pd.read_csv('fashion-mnist_test.csv')
    data = pd.concat([data1, data2])
    X = data[data['label'] == 8]
    y = X[['label']].values
    X = X.drop(columns=['label']).values / 255

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=0.85)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val, y_val, train_size=0.8)

    network = network.train(X_train, X_train, X_val, X_val, learning_rate=0.00005, epochs=250)

    y_pred = network.feed_forward(X_test)

    print("Loss of the best model:")
    print(network.calc_loss(y_pred, X_test).mean())

    convert_to_image(y_pred)


def visualise_first_hidden():
    with open('nn_bags', 'rb') as f:
        network = pickle.load(f)
    first_layer_weights = np.array([neuron.weights for neuron in network.layers[0]])

    def plot_neurons(start_index, end_index, figure_number):
        plt.figure(figure_number, figsize=(15, 15))
        for i in range(start_index, end_index):
            weights = first_layer_weights[i]
            normalized_input = (weights - weights.min()) / (weights.max() - weights.min())
            image = normalized_input.reshape(28, 28)

            ax = plt.subplot(8, 8, i - start_index + 1)
            ax.imshow(image, cmap='gray')
            ax.axis('off')
            ax.set_aspect('equal')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.show()

    plot_neurons(0, 32, 1)
    plot_neurons(32, 64, 2)
    plot_neurons(64, 96, 3)
    plot_neurons(96, 128, 4)


if __name__ == "__main__":
    visualize()
