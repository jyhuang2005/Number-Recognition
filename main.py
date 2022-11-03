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
    count = 0
    for img in train_images:
        grayscale_array = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                grayscale_array.append([img[r][c] / 255])
        arr.append(np.array(grayscale_array))
        count += 1
        if count == 20:
            break
    return arr


def mserror(actual_values, predicted_values):
    return mean_squared_error(actual_values, predicted_values, squared=True)


def MSE(index):
    # for i in l3.get_matrix()[0]:
    #     print(i.get_value(), i.get_weights(), i.get_bias())

    actuals = []
    predicts = []

    matrix = l3.matrix
    for j in range(len(matrix)):
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
    matrix = l3.matrix

    for val in range(len(matrix)):
        if val == int(train_labels[index]):
            dacost.append((matrix[val, 0] - 1) * 2)
        else:
            dacost.append(matrix[val, 0] * 2)


l1 = la.Layer(16)
l2 = la.Layer(16, l1)
l3 = la.Layer(10, l2)

prop_c = 20.0


for j in range(100):
    total = 0
    l3_wshifts = np.empty([l3.num_neurons, l3.prev_len])
    l3_wshifts[:] = 0
    l3_bshifts = np.empty([l3.num_neurons, 1])
    l3_bshifts[:] = 0

    for i in range(len(vect_arr)):
        l1.update(vect_arr[i])
        l2.update()
        l3.update()
        total += MSE(i)

        dacost = []
        d_a_to_cost(i)
        dza1 = dsigmoid(l1.prod)
        dza2 = dsigmoid(l2.prod)
        dza3 = dsigmoid(l3.prod)

        dzcost = []

        prev_matrix = l2.matrix
        for m in range(l3.num_neurons):
            dzcost1 = dacost[m] * dza3[m, 0]
            dzcost.append(dzcost1)
            for n in range(l3.prev_len):
                # print(dacost[m], dza3[m, 0], prev_matrix[n, 0])
                # print(-dacost[m] * dza3[m, 0] * prev_matrix[n, 0])
                l3_wshifts[m, n] -= dzcost1 * prev_matrix[n, 0]
            l3_bshifts[m, 0] -= dzcost1

        # for m in range(l2.num_neurons):
        #

    avg = total / len(vect_arr)
    print(10 * avg)

    l3_wshifts /= len(vect_arr)
    l3_wshifts *= prop_c
    l3_bshifts /= len(vect_arr)
    l3_bshifts *= prop_c


    l3.change_weights(l3_wshifts)
    l3.change_biases(l3_bshifts)

correct = 0

for i in range(11, 20):
    l1.update(vect_arr[i])
    l2.update()
    l3.update()
    total += MSE(i)
    maxim = 0
    maxim_index = 0
    for j in range(len(l3.matrix)):
        if l3.matrix[j, 0] > maxim:
            maxim = l3.matrix[j, 0]
            maxim_index = j
    print(maxim_index)
    print(train_labels[i])
    if maxim_index == train_labels[i]:
        correct += 1
print(correct)




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
