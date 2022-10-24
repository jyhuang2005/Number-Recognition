import layer

print('Hello World')
print('test')

# added comment
# justin's contribution!


class Neuron:
    def __init__(self, value=0, connects=None):
        if connects is None:
            connects = []
        self.value = value
        for other_neuron in connects:
            self.value += other_neuron.get_value()

    def get_value(self):
        return self.value


n1 = Neuron(value=5)
n2 = Neuron(value=6)
n3 = Neuron(connects=[n1, n2])
print(n1.get_value(), n2.get_value(), n3.get_value())

l1 = layer.Layer(3)
print(l1.get_num_neurons())
