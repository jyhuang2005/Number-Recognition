import layer as la
import neuron as n

print('Hello World')
print('test')

# added comment
# justin's contribution!

n1 = n.Neuron(value=5)
n2 = n.Neuron(value=6)
n3 = n.Neuron(connects=[n1, n2])
print(n1.get_value(), n2.get_value(), n3.get_value())

l1 = la.Layer(3)
print(l1.get_num_neurons())
