import math

import idx2numpy
import layer as la
import neuron as n
from sklearn.metrics import mean_squared_error

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


def calculate_mean_squared_error(actuals, predicts):
    return mean_squared_error(actuals, predicts, squared=False)


l1 = la.Layer(None, val_array=train_images[0])
l2 = la.Layer(16, l1)
l3 = la.Layer(16, l2)
l4 = la.Layer(10, l3)

for i in l1.get_neuron_array():
    for j in i:
        print(j.get_value())

for i in l2.get_neuron_array()[0]:
    print(i.get_value(), i.get_weights(), i.get_bias())

for i in l3.get_neuron_array()[0]:
    print(i.get_value(), i.get_weights(), i.get_bias())

for i in l4.get_neuron_array()[0]:
    print(i.get_value(), i.get_weights(), i.get_bias())


# for i in range(1, len(train_images)):
#     l1.update(train_images[i])
#     l2.update()
#     l3.update()
#     l4.update()
#     print(i)

for i in l2.get_neuron_array()[0]:
    print(i.get_value(), i.get_weights(), i.get_bias())

for i in l3.get_neuron_array()[0]:
    print(i.get_value(), i.get_weights(), i.get_bias())

for i in l4.get_neuron_array()[0]:
    print(i.get_value(), i.get_weights(), i.get_bias())

print()

actuals = []
predicts = []

for i in l4.get_neuron_array()[0]:
    if i == int(train_labels[0]):
        actuals.append(0)
    else:
        actuals.append(1)
    predicts.append(i.get_value())

print(calculate_mean_squared_error(actuals, predicts))

