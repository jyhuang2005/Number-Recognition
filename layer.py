import neuron as n


class Layer:
    def __init__(self, num_neurons, prev_layer=None, val_array=None):
        self.num_neurons = num_neurons
        self.neuron_array = []
        if prev_layer is None:
            for i in val_array:
                self.neuron_array.append(n.Neuron(value=i))
        if val_array is None:
            for i in range(0, num_neurons):
                self.neuron_array.append(n.Neuron(connects=prev_layer.get_neuron_array()))

    def get_num_neurons(self):
        return self.num_neurons

    def get_neuron_array(self):
        return self.neuron_array

    def update(self, val_array=None):
        if val_array is None:
            for nr in self.neuron_array:
                nr.update()
        else:
            for i in range(0, len(val_array)):
                self.neuron_array[i].update(val_array[i])

