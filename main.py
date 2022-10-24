import layer as la
import neuron as n


l1 = la.Layer(5, val_array=[0, 0.2, 0.5])
l2 = la.Layer(3, l1)
l3 = la.Layer(5, l2)

for i in l1.get_neuron_array():
    print(i.get_value())
print(l1.get_num_neurons())

for i in l2.get_neuron_array():
    print(i.get_value(), i.get_weights(), i.get_bias())
print(l2.get_num_neurons())

for i in l3.get_neuron_array():
    print(i.get_value(), i.get_weights(), i.get_bias())
print(l3.get_num_neurons())

