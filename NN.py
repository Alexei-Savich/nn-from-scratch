import numpy as np
from matplotlib import pyplot as plt
import pickle
import umap
from sklearn.decomposition import PCA


class Neuron:

    def __init__(self, input_size: int, activation: str, weights_strategy: str = None):
        self.input_size = input_size
        self.activation = activation
        if weights_strategy == 'xavier':
            lower, upper = -(1.0 / np.sqrt(input_size)), (1.0 / np.sqrt(input_size))
            self.weights = lower + np.random.rand(input_size) * (upper - lower)
            self.bias = lower + np.random.rand(1) * (upper - lower)
        elif weights_strategy == 'he':
            std = np.sqrt(2.0 / input_size)
            self.weights = np.random.randn(input_size) * std
            self.bias = np.random.randn(1) * std
        else:
            self.weights = np.random.rand(input_size)
            self.bias = np.random.rand(1)


class NeuralNetwork:

    def __init__(self, layer_sizes: list[int], activations: list[str], weights_strategy: str = 'xavier'):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.best_nn = None
        self.best_loss = None
        self.layers = []

        for i in range(1, len(layer_sizes)):
            self.layers.append(
                [Neuron(layer_sizes[i - 1], activations[i], weights_strategy) for _ in range(layer_sizes[i])])

    def apply_activations(self, X, activation_function, derivative=False):
        if activation_function == 'relu':
            if derivative:
                return np.where(X > 0, 1, 0)
            else:
                return np.maximum(0, X)
        elif activation_function == 'sigmoid':
            if derivative:
                sig = 1 / (1 + np.exp(-X))
                return sig * (1 - sig)
            else:
                return 1 / (1 + np.exp(-X))
        elif activation_function == 'softmax':
            if derivative:
                exp_X = np.exp(X)
                softmax = exp_X / exp_X.sum(axis=1, keepdims=True)
                return softmax * (1 - softmax)
            else:
                return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
        elif activation_function == 'linear':
            if derivative:
                return np.ones_like(X)
            else:
                return X
        else:
            raise RuntimeError(f'Incorrect activation function: {activation_function}')

    def calc_loss(self, y_pred, y):
        last_activation = self.activations[-1]
        if last_activation == 'sigmoid':
            return -1 * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean(axis=1)
        elif last_activation in ['relu', 'linear']:
            return ((y_pred - y) ** 2).mean()
        elif last_activation == 'softmax':
            return -1 * y * np.log(y_pred)
        else:
            raise RuntimeError(f'Unsupported output activation function: {last_activation}')

    def feed_forward(self, X):
        self.layer_outputs = []
        self.weighted_sums = []
        curr_input = X
        for layer in self.layers:
            curr_input = np.dot(curr_input, np.stack([neuron.weights for neuron in layer], axis=1))
            curr_input = curr_input + np.reshape(np.array([neuron.bias for neuron in layer]), (1, -1))
            self.weighted_sums.append(curr_input)
            curr_input = self.apply_activations(curr_input, layer[0].activation)
            self.layer_outputs.append(curr_input)
        return curr_input

    def back_propagation(self, X, y, learning_rate):
        last_activation = self.activations[-1]
        if last_activation == 'sigmoid':
            dA = - (np.divide(y, self.layer_outputs[-1]) - np.divide(1 - y, 1 - self.layer_outputs[-1]))
        elif last_activation == 'softmax':
            dA = - (y - self.layer_outputs[-1])
        elif last_activation == 'linear':
            dA = 2 * (self.layer_outputs[-1] - y)
        else:
            raise RuntimeError(f'Unsupported output activation function: {last_activation}')
        # print(f'Mean First dA of FCN: {(dA ** 2).mean()}')

        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]
            activation_function = self.activations[i + 1]

            dZ = dA * self.apply_activations(self.weighted_sums[i], activation_function, derivative=True)
            dW = np.dot(self.layer_outputs[i - 1].T if i > 0 else X.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)

            for j, neuron in enumerate(current_layer):
                neuron.weights -= learning_rate * dW[:, j]
                neuron.bias -= learning_rate * db[:, j]

            dA = np.dot(dZ, np.array([neuron.weights for neuron in current_layer]))

        # print(f'Mean dA of FCN: {(dA ** 2).mean()}')
        return dA

    def train(self, X_train, y_train, X_val, y_val, learning_rate: float, epochs: int, batch_size: int = 32,
              early_stop_epochs: int = 0):
        epoch_losses = []
        m = X_train.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            for i in range(0, m, batch_size):
                X_mini = X_shuffled[i:i + batch_size]
                y_mini = y_shuffled[i:i + batch_size]
                self.feed_forward(X_mini)
                self.back_propagation(X_mini, y_mini, learning_rate)

            if self.activations[-1] == 'sigmoid':
                loss = self.calc_loss(self.feed_forward(X_val), y_val).mean()
            elif self.activations[-1] == 'linear':
                loss = self.calc_loss(self.feed_forward(X_val), y_val)
            elif len(self.layers[-1]) > 1:
                loss = np.average(np.sum(self.calc_loss(self.feed_forward(X_val), y_val), axis=1))
            else:
                loss = self.calc_loss(self.feed_forward(X_val), y_val)

            stop_now = False
            if early_stop_epochs and len(epoch_losses) > early_stop_epochs:
                stop_now = True
                for prev_loss in epoch_losses[-1 - early_stop_epochs:]:
                    if self.best_loss >= prev_loss:
                        stop_now = False
                        break

            epoch_losses.append(loss)

            if stop_now:
                print(
                    f'Stopping early at epoch {epoch} because best loss {self.best_loss} did not descrease over last {early_stop_epochs} iterations: {epoch_losses[-2 - early_stop_epochs:]}')
                break

            if self.best_loss is None or self.best_loss > loss:
                self.best_nn = self.copy()
                self.best_loss = loss

            print(f'Loss after epoch {epoch}: {loss}')

        with open('nn', 'wb') as file:
            pickle.dump(self.best_nn, file)

        plt.plot(range(len(epoch_losses)), epoch_losses)
        plt.title('Average loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        return self.best_nn

    def copy(self):
        copied_nn = NeuralNetwork(self.layer_sizes, self.activations)
        copied_nn.layers = []
        for layer in self.layers:
            copied_layer = []
            for neuron in layer:
                copied_neuron = Neuron(neuron.input_size, neuron.activation)
                copied_neuron.weights = np.copy(neuron.weights)
                copied_neuron.bias = np.copy(neuron.bias)
                copied_layer.append(copied_neuron)
            copied_nn.layers.append(copied_layer)
        copied_nn.best_loss = self.best_loss
        return copied_nn

    def visualize_umap(self, data, labels, layer_name):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)

        unique_labels = set(labels)
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embedding[indices, 0], embedding[indices, 1], label=label, cmap='Spectral', s=5)

        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'UMAP projection of the Autoencoder Features of layer {layer_name}', fontsize=12)
        plt.legend()
        plt.show()

    def get_hidden_layer_outputs(self, X):
        self.feed_forward(X)
        return [layer_output for layer_output in self.layer_outputs[:-1]]

    def visualize_hidden_layers(self, X):
        hidden_layers_outputs = self.get_hidden_layer_outputs(X)
        for i, layer_output in enumerate(hidden_layers_outputs):
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(layer_output)

            plt.figure(figsize=(8, 6))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, alpha=0.7)
            plt.title(f'Hidden Layer {i + 1} Feature Representation (2D PCA)')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.show()
