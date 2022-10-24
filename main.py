class Neuron:
    def __init__(self, value=0, connects=None):
        if connects is None:
            connects = []
        self.value = value
        for other_neuron in connects:
            self.value += other_neuron.get_value()

    def get_value(self):
        return self.value


class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons


class Brain:
    def __init__(self, num_layers, num_neurons):
        self.num_layers = num_layers
        self.num_neurons = num_neurons


test = Brain(4, [784, 20, 20, 10])

n1 = Neuron(value=5)
n2 = Neuron(value=6)
n3 = Neuron(connects=[n1, n2])
print(n1.get_value(), n2.get_value(), n3.get_value())
