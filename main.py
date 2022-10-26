import idx2numpy
import layer as la
import neuron as n


train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")


def create_grayscale_array():
    arr = []
    for img in train_images:
        grayscale_array = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                grayscale_array.append(img[r][c] / 255)
        arr.append(grayscale_array)
    return arr


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

print(create_grayscale_array())
