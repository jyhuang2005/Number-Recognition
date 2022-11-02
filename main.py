import idx2numpy
import numpy as np
import layer as la
import neuron as n
from sklearn.metrics import mean_squared_error

train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")

def dsigmoid(x):
    return np.exp(x)/np.power((1+np.exp(x)), 2)

def create_grayscale_vector_array():
    arr = []
    for img in train_images:
        grayscale_array = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                grayscale_array.append([img[r][c] / 255])
        arr.append(np.array(grayscale_array))
        break
    return arr


def mserror(actual_values, predicted_values):
    return mean_squared_error(actual_values, predicted_values, squared=True)


def MSE(index):
    # for i in l3.get_matrix()[0]:
    #     print(i.get_value(), i.get_weights(), i.get_bias())

    actuals = []
    predicts = []

    matrix = l3.get_matrix()
    for j in range(0, len(matrix)):
        val = matrix[j, 0]
        if j == int(train_labels[index]):
            actuals.append(1)
        else:
            actuals.append(0)
        predicts.append(val)

    return mserror(actuals, predicts)


vect_arr = create_grayscale_vector_array()

# def top_weights:
#     # 2 * w * dsigmoid(z) * (a - y)
#
# def propagate_weights:


def d_a_to_cost(index):
    matrix = l3.get_matrix()

    for i in range(0, len(matrix)):
        if i == int(train_labels[index]):
            dacost.append((matrix[i, 0] - 1) * 2)
        else:
            dacost.append(matrix[i, 0] * 2)

for j in range(0, 20):
    l1 = la.Layer(16)
    l2 = la.Layer(16, l1)
    l3 = la.Layer(10, l2)

    total = 0
    for i in range(0, len(vect_arr)):
        l1.update(vect_arr[i])
        l2.update()
        l3.update()
        total += MSE(j)

    dacost = []
    d_a_to_cost(j)
    print(dacost)
    dza1 = dsigmoid(l1.get_prod())
    dza2 = dsigmoid(l2.get_prod())
    dza3 = dsigmoid(l3.get_prod())

    l3_shifts = np.empty([l3.num_neurons, l3.prev_len])
    for m in range(l3.num_neurons):
        for n in range(l3.prev_len):
            l3_shifts[m, n] = dacost[m] * dza3[m] * l2.get_matrix()[n, 0]
    l3.change_weights(l3_shifts)



    avg = total / len(vect_arr)
    print(10*avg)










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
