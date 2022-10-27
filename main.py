import idx2numpy
import numpy as np
import layer as la
import neuron as n
from sklearn.metrics import mean_squared_error

train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")


def create_grayscale_vector_array():
    arr = []
    for img in train_images:
        grayscale_array = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                grayscale_array.append([img[r][c] / 255])
        arr.append(np.array(grayscale_array))
    return arr


def root_mean_squared_error(actual_values, predicted_values):
    return mean_squared_error(actual_values, predicted_values, squared=False)


def RMSE():
    # for i in l3.get_matrix()[0]:
    #     print(i.get_value(), i.get_weights(), i.get_bias())

    actuals = []
    predicts = []

    matrix = l3.get_matrix()
    for j in range(0, len(matrix)):
        val = matrix[j, 0]
        if j == int(train_labels[0]):
            actuals.append(1)
        else:
            actuals.append(0)
        predicts.append(val)

    return root_mean_squared_error(actuals, predicts)


vect_arr = create_grayscale_vector_array()

for j in range(0, 20):
    l1 = la.Layer(16)
    l2 = la.Layer(16, l1)
    l3 = la.Layer(10, l2)

    total = 0
    for i in range(0, len(vect_arr)):
        l1.update(vect_arr[i])
        l2.update()
        l3.update()
        total += RMSE()

    avg = total / len(vect_arr)
    print(avg)

# l1 = la.Layer(None, val_array=train_images[0])
# l2 = la.Layer(16, l1)
# l3 = la.Layer(16, l2)
# l4 = la.Layer(10, l3)
#
# for i in l1.get_neuron_array():
#     for j in i:
#         print(j.get_value())
#
# for i in l2.get_neuron_array()[0]:
#     print(i.get_value(), i.get_weights(), i.get_bias())
#
# for i in l3.get_neuron_array()[0]:
#     print(i.get_value(), i.get_weights(), i.get_bias())
#
# for i in l4.get_neuron_array()[0]:
#     print(i.get_value(), i.get_weights(), i.get_bias())
#
#
# for i in range(1, len(train_images)):
#     l1.update(train_images[i])
#     l2.update()
#     l3.update()
#     l4.update()
#     print(i)
#
# for i in l2.get_neuron_array()[0]:
#     print(i.get_value(), i.get_weights(), i.get_bias())
#
# for i in l3.get_neuron_array()[0]:
#     print(i.get_value(), i.get_weights(), i.get_bias())
#
# for i in l4.get_neuron_array()[0]:
#     print(i.get_value(), i.get_weights(), i.get_bias())
