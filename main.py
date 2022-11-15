import idx2numpy
import numpy as np
import layer as la
import neuron as n
from sklearn.metrics import mean_squared_error

train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")

test_images = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

set_size = 100
set_num = 60000//set_size


def dsigmoid(x):
    return np.exp(x)/np.power((1+np.exp(x)), 2)


def create_grayscale_vector_array(image_set):
    arr = []
    count = 0
    for p in range(len(image_set) // set_size):
        arr.append([])
    for img in image_set:
        grayscale_array = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                grayscale_array.append([img[r][c] / 255])
        arr[count // set_size].append(np.array(grayscale_array))
        count += 1
        # if count == 200:
        #     break
    return arr


def mserror(actual_values, predicted_values):
    return mean_squared_error(actual_values, predicted_values, squared=True)


def MSE(index):
    # for i in l3.get_matrix()[0]:
    #     print(i.get_value(), i.get_weights(), i.get_bias())

    actuals = []
    predicts = []

    temp_matrix = l3.matrix
    for k in range(len(temp_matrix)):
        val = temp_matrix[k, 0]
        if k == int(train_labels[index]):
            actuals.append(1)
        else:
            actuals.append(0)
        predicts.append(val)

    return mserror(actuals, predicts)


train_vect_arr = create_grayscale_vector_array(train_images)
test_vect_arr = create_grayscale_vector_array(test_images)

# def top_weights:
#     # 2 * w * dsigmoid(z) * (a - y)
#
# def propagate_weights:


def d_a_to_cost(index):
    temp_matrix = l3.matrix

    for val in range(len(temp_matrix)):
        if val == int(train_labels[index]):
            da3cost.append((temp_matrix[val, 0] - 1) * 2)
        else:
            da3cost.append(temp_matrix[val, 0] * 2)


def update_text_files():
    with open('percentcorrect.txt', 'w') as f:
        f.write(str(percent_correct))
    np.savetxt("l1weights.txt", l1.weights)
    np.savetxt("l2weights.txt", l2.weights)
    np.savetxt("l3weights.txt", l3.weights)
    np.savetxt("l1biases.txt", l1.biases)
    np.savetxt("l2biases.txt", l2.biases)
    np.savetxt("l3biases.txt", l3.biases)


def get_weights(layer_num):
    if layer_num == 1:
        return np.loadtxt("l1weights.txt")
    elif layer_num == 2:
        return np.loadtxt("l2weights.txt")
    elif layer_num == 3:
        return np.loadtxt("l3weights.txt")


def get_biases(layer_num):
    if layer_num == 1:
        return np.loadtxt("l1biases.txt")
    elif layer_num == 2:
        return np.loadtxt("l2biases.txt")
    elif layer_num == 3:
        return np.loadtxt("l3biases.txt")


l1 = la.Layer(100, weights=get_weights(1), biases=np.rot90([get_biases(1)], 3))
l2 = la.Layer(100, l1, weights=get_weights(2), biases=np.rot90([get_biases(2)], 3))
l3 = la.Layer(10, l2, weights=get_weights(3), biases=np.rot90([get_biases(3)], 3))

prop_c = 2.0


for j in range(600):
    train_vect = train_vect_arr[j % set_num]
    total = 0
    l3_wshifts = np.empty([l3.num_neurons, l3.prev_len])
    l3_wshifts.fill(0)
    l3_bshifts = np.empty([l3.num_neurons, 1])
    l3_bshifts.fill(0)
    l2_wshifts = np.empty([l2.num_neurons, l2.prev_len])
    l2_wshifts.fill(0)
    l2_bshifts = np.empty([l2.num_neurons, 1])
    l2_bshifts.fill(0)
    l1_wshifts = np.empty([l1.num_neurons, l1.prev_len])
    l1_wshifts.fill(0)
    l1_bshifts = np.empty([l1.num_neurons, 1])
    l1_bshifts.fill(0)

    for i in range(len(train_vect)):
        l1.update(train_vect[i])
        l2.update()
        l3.update()
        total += MSE((j % set_num) * set_size + i)

        da3cost = []
        da2cost = []
        da1cost = []
        for k in range(l1.num_neurons):
            da1cost.append(0)
        for k in range(l2.num_neurons):
            da2cost.append(0)
        d_a_to_cost((j % set_num) * set_size + i)
        dza1 = dsigmoid(l1.prod)
        dza2 = dsigmoid(l2.prod)
        dza3 = dsigmoid(l3.prod)

        layer = l3
        matrix = l3.matrix
        prev_matrix = layer.prev_matrix
        for m in range(layer.num_neurons):
            dz3costm = da3cost[m] * dza3[m, 0]
            for n in range(layer.prev_len):
                l3_wshifts[m, n] -= dz3costm * prev_matrix[n, 0]
                da2cost[n] += dz3costm * layer.weights[m, n]
            l3_bshifts[m, 0] -= dz3costm

        layer = l2
        matrix = prev_matrix
        prev_matrix = layer.prev_matrix
        for m in range(l2.num_neurons):
            dz2costm = da2cost[m] * dza2[m, 0]
            for n in range(l2.prev_len):
                l2_wshifts[m, n] -= dz2costm * prev_matrix[n, 0]
                da1cost[n] += dz2costm * layer.weights[m, n]
            l2_bshifts[m, 0] -= dz2costm

        layer = l1
        matrix = prev_matrix
        prev_matrix = layer.prev_matrix
        for m in range(l1.num_neurons):
            dz1costm = da1cost[m] * dza1[m, 0]
            for n in range(l1.prev_len):
                l1_wshifts[m, n] -= dz1costm * prev_matrix[n, 0]
            l1_bshifts[m, 0] -= dz1costm

    avg = total / len(train_vect)
    # print(10 * avg)
    if j % 10 == 0:
        print(f'{j} {10 * avg}')

    l3_wshifts /= len(train_vect)
    l3_wshifts *= prop_c
    l3_bshifts /= len(train_vect)
    l3_bshifts *= prop_c
    l2_wshifts /= len(train_vect)
    l2_wshifts *= prop_c
    l2_bshifts /= len(train_vect)
    l2_bshifts *= prop_c
    l1_wshifts /= len(train_vect)
    l1_wshifts *= prop_c
    l1_bshifts /= len(train_vect)
    l1_bshifts *= prop_c

    l3.change_weights(l3_wshifts)
    l3.change_biases(l3_bshifts)
    l2.change_weights(l2_wshifts)
    l2.change_biases(l2_bshifts)
    l1.change_weights(l1_wshifts)
    l1.change_biases(l1_bshifts)

correct = 0
tot = 0

for i in range(10000):
    l1.update(test_vect_arr[i // set_size][i % set_size])
    l2.update()
    l3.update()
    maxim = 0
    maxim_index = 0
    for j in range(len(l3.matrix)):
        if l3.matrix[j, 0] > maxim:
            maxim = l3.matrix[j, 0]
            maxim_index = j
    print(maxim_index)
    print(test_labels[i])
    if maxim_index == test_labels[i]:
        correct += 1
    tot += 1
percent_correct = correct/tot
if percent_correct > float(np.loadtxt("percentcorrect.txt")) and tot == 10000:
    update_text_files()
# if tot == 10000:
#     update_text_files()

print(f'{correct} / {tot}')
print(np.rot90([get_biases(1)], 3))

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
