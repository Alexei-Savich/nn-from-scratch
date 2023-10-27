import pickle

import numpy as np
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd

import NN as nn


def generate_data(num_samples: int, num_features: int):
    X = np.random.randint(-10, 11, size=(num_samples, num_features))

    zero_sum_size = int(num_samples * 0.3)
    X_zero_sum = np.zeros((zero_sum_size, num_features))
    for i in range(zero_sum_size):
        X_zero_sum[i, :-1] = np.random.randint(-10, 11, num_features - 1)
        X_zero_sum[i, -1] = -np.sum(X_zero_sum[i, :-1])
    X = np.vstack((X, X_zero_sum))
    np.random.shuffle(X)
    y_softmax = np.zeros((X.shape[0], 3))
    X_sum = np.sum(X, axis=1)
    y_softmax[X_sum > 0] = [1, 0, 0]
    y_softmax[X_sum < 0] = [0, 1, 0]
    y_softmax[X_sum == 0] = [0, 0, 1]
    return X, y_softmax


def test_arbitrary():
    network = nn.NeuralNetwork([5, 15, 15, 3], [None, 'relu', 'sigmoid', 'softmax'], weights_strategy='he')
    X, y = generate_data(1000, 5)

    X_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=0.9)

    network = network.train(X_train, y_train, x_val, y_val, learning_rate=0.1, epochs=1000, early_stop_epochs=100)

    X_test, y_test = generate_data(100, 5)
    y_pred = network.feed_forward(X_test)
    sum_test = np.sum(X_test, axis=1)
    for x, y in zip(sum_test, y_pred):
        print(f"Input sum: {x}, y_pred: {np.round(y)}")

    print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))


def test_mnist(file_name, load_from_disk=False):
    if load_from_disk:
        with open(file_name, 'rb') as file:
            network = pickle.load(file)
    else:
        network = nn.NeuralNetwork([784, 100, 100, 100, 10], [None, 'sigmoid', 'relu', 'sigmoid', 'softmax'],
                                   weights_strategy='he')
    data = pd.read_csv('mnist.csv')
    X = data.drop(columns=['label']).values
    y = data[['label']].values

    num_classes = 10
    one_hot_encoded = np.zeros((len(y), num_classes))
    for i, item in enumerate(y):
        one_hot_encoded[i, item[0]] = 1
    y = one_hot_encoded

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=0.8)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val, y_val, train_size=0.5)

    if not load_from_disk:
        network = network.train(X_train, y_train, X_val, y_val, learning_rate=0.005, epochs=200, early_stop_epochs=0)

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


def test_insurance(file_name, load_from_disk):
    if load_from_disk:
        with open(file_name, 'rb') as file:
            network = pickle.load(file)
    else:
        network = nn.NeuralNetwork([8, 100, 20, 1], [None, 'sigmoid', 'sigmoid', 'linear'], weights_strategy='xavier')
    df = pd.read_csv('insurance.csv')

    df[["sex"]] = df[["sex"]].map(lambda x: 1 if x == "male" else 0)
    df[["smoker"]] = df[["smoker"]].map(lambda x: 1 if x == "yes" else 0)
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    X = df.drop('charges', axis=1).map(lambda element: float(element)).values
    y = df[['charges']].map(lambda element: float(element) / 1000).values

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=0.8)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val, y_val, train_size=0.5)

    if not load_from_disk:
        network = network.train(X_train, y_train, X_val, y_val, learning_rate=0.000002, epochs=10000,
                                early_stop_epochs=0)

    y_pred = network.feed_forward(X_test)

    for pred, test in zip(y_pred, y_test):
        print(f'Test: {test}; pred: {pred}, pred - test {pred - test}; 1 - pred/test: {1 - pred / test}')

    print("Mean y:")
    print(np.mean(y_test))
    print("Mean y_pred:")
    print(np.mean(y_pred))
    print("Loss of the best NN:")
    print(network.calc_loss(y_pred, y_test))
    print("Mean |pred - test|:")
    print(np.mean(np.abs(y_pred - y_test)))
    print("Mean |1 - pred/test|:")
    print(np.mean(np.abs(1 - (y_pred / y_test))))


if __name__ == '__main__':
    # test_mnist(load_from_disk=False, file_name='mnist_86_percent')
    test_insurance(load_from_disk=False, file_name='nn')
