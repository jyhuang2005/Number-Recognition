import layer as la
import neuron as n

n1 = n.Neuron(value=5)
n2 = n.Neuron(value=6)
n3 = n.Neuron(connects=[n1, n2])
print(n1.get_value(), n2.get_value(), n3.get_value())

l1 = la.Layer(5, val_array=[1, 2, 3, 4, 5])
l2 = la.Layer(3, l1)
l3 = la.Layer(5, l2)

for i in l1.get_neuron_array():
    print(i.get_value())
print(l1.get_num_neurons())

for i in l2.get_neuron_array():
    print(i.get_value())
print(l2.get_num_neurons())

for i in l3.get_neuron_array():
    print(i.get_value())
print(l3.get_num_neurons())
