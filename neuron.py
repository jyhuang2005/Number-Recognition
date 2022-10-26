import random as r
import math as m


def sigmoid(val):
    val2 = 1 / (1 + m.exp(-val))
    return val2


class Neuron:
    def __init__(self, value=0, connects=None):
        self.value = value
        if connects is None:
            self.connects = []
        else:
            self.connects = connects
            self.weights = []
            for i in range(0, len(self.connects)):
                self.weights.append([])
                for j in range(0, len(self.connects[0])):
                    self.weights[i].append(2*r.random() - 1)
            self.bias = 2*r.random() - 1
            print(self.weights)

            for i in range(0, len(self.connects)):
                for j in range(0, len(self.connects[0])):
                    self.value += self.connects[i][j].get_value()*self.weights[i][j]
            self.value += self.bias
            self.value = sigmoid(self.value)

    def get_value(self):
        return self.value

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, val=None):
        if val is None:
            self.value = 0
            for i in range(0, len(self.connects)):
                for j in range(0, len(self.connects[0])):
                    self.value += self.connects[i][j].get_value()*self.weights[i][j]
            self.value += self.bias
            self.value = sigmoid(self.value)
        else:
            self.value = val
