print('Hello World')
print('test')

# added comment
# justin's contribution!

class Neuron:
    def __init__(self, value):
        self.value = value

    def __init__(self, connects):
        self.value = 0
        for other_neuron in connects:
            self.value += other_neuron.get_value()

    def get_value(self):
        return self.get_value()


class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons


class Brain:
    def __init__(self, num_layers, num_neurons):
        self.num_layers = num_layers
        self.num_neurons = num_neurons


test = Brain(4, [784, 20, 20, 10])


