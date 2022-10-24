class Neuron:
    def __init__(self, value=0, connects=None):
        if connects is None:
            connects = []
        self.value = value
        for other_neuron in connects:
            self.value += other_neuron.get_value()

    def get_value(self):
        return self.value
