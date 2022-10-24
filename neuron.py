import random as r


class Neuron:
    def __init__(self, value=0, connects=None):
        if connects is None:
            connects = []
        self.value = value
        self.weights = []
        for i in range(0, len(connects)):
            self.weights.append(r.random())
        for other_neuron in connects:
            self.value += other_neuron.get_value()

    def get_value(self):
        return self.value

    def get_weights(self):
        return self.weights
