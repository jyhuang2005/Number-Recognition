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
            for i in range(0, len(connects)):
                self.weights.append(2*r.random() - 1)
            self.bias = 2*r.random() - 1

            for i in range(0, len(connects)):
                self.value += self.connects[i].get_value()*self.weights[i]
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
                self.value += self.connects[i].get_value()*self.weights[i]
            self.value += self.bias
            self.value = sigmoid(self.value)
        else:
            self.value = val
