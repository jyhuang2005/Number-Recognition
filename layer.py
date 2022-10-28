import neuron as n
import numpy as np
import random as r


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return np.exp(x)/np.power((1+np.exp(x)), 2)

print(dsigmoid(np.array([2])))

class Layer:
    def __init__(self, num_neurons, prev_layer=None):
        self.num_neurons = num_neurons
        # self.neuron_array = []
        self.prev_len = 784
        if prev_layer is not None:
            self.prev_layer = prev_layer
            self.prev_len = len(prev_layer)

        self.prev_matrix = None

        self.weights = np.empty([self.num_neurons, self.prev_len])
        for i in range(0, num_neurons):
            for j in range(0, self.prev_len):
                self.weights[i, j] = 2*r.random() - 1

        self.biases = np.empty([num_neurons, 1])
        for i in range(0, num_neurons):
            self.biases[i, 0] = -2*r.random()

        self.matrix = np.array([1, num_neurons])

        # for i in range(0, num_neurons):
        #     self.neuron_array[0].append(n.Neuron(connects=prev_layer.get_neuron_array()))

    def __len__(self):
        return self.num_neurons

    def get_num_neurons(self):
        return self.num_neurons

    def get_matrix(self):
        return self.matrix

    def get_weights(self):
        return self.weights

    # def get_neuron_array(self):
    #     return self.neuron_array

    # def update(self, val_array=None):
    #     if val_array is None:
    #         for nr in self.neuron_array[0]:
    #             nr.update()
    #     else:
    #         for i in range(0, len(val_array)):
    #             for j in range(0, len(val_array[0])):
    #                 self.neuron_array[i][j].update(val_array[i][j]/255)

    def update(self, prev_matrix=None):
        if prev_matrix is None:
            self.prev_matrix = self.prev_layer.get_matrix()
        else:
            self.prev_matrix = prev_matrix

        prod = np.matmul(self.weights, self.prev_matrix)
        self.matrix = sigmoid(np.add(prod, self.biases))

