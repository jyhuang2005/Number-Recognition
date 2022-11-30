import idx2numpy
import numpy as np
import layer as la
import emnist
from sklearn.metrics import mean_squared_error


# train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
# train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")

# test_images = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
# test_labels = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

train_images, train_labels = emnist.extract_training_samples('balanced')
test_images, test_labels = emnist.extract_test_samples('balanced')

set_size = 100
set_num = len(train_images) // set_size
test_size = len(test_images)
prefix = "emnist_"


def dsigmoid(x):
    return np.exp(x) / np.power((1 + np.exp(x)), 2)


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


def mse(index):
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

    return mean_squared_error(actuals, predicts, squared=True)


def create_one_dimensional(arr):
    return np.array(arr).ravel()


def create_reshaped_vector_array(arr):
    return np.reshape(arr, (set_num, set_size, 784, 1))


def write_train_data_to_file():
    train_arr = create_grayscale_vector_array(train_images)

    train_data = np.reshape(create_one_dimensional(train_arr), (set_num, set_size, 784, 1))

    with open(prefix + 'train_vect_arr.txt', 'w') as outfile:
        for threeD_data_slice in train_data:
            for twoD_data_slice in threeD_data_slice:
                np.savetxt(outfile, twoD_data_slice, fmt='%-7.8f')
                outfile.write('\n')


def write_test_data_to_file():
    test_arr = create_grayscale_vector_array(test_images)

    test_data = np.reshape(create_one_dimensional(test_arr), (test_size // set_size, set_size, 784, 1))

    with open(prefix + 'test_vect_arr.txt', 'w') as outfile:
        for threeD_data_slice in test_data:
            for twoD_data_slice in threeD_data_slice:
                np.savetxt(outfile, twoD_data_slice, fmt='%-7.8f')
                outfile.write('\n')


# write_train_data_to_file()  # get rid of these when run once
# write_test_data_to_file()


def get_train_vect_arr():
    return np.loadtxt(prefix + 'train_vect_arr.txt').reshape((set_num, set_size, 784, 1))


def get_test_vect_arr():
    return np.loadtxt(prefix + 'test_vect_arr.txt').reshape((test_size // set_size, set_size, 784, 1))


train_vect_arr = get_train_vect_arr()
test_vect_arr = get_test_vect_arr()


def d_a_to_cost(index):
    temp_matrix = l3.matrix

    for val in range(len(temp_matrix)):
        if val == int(train_labels[index]):
            da3cost.append((temp_matrix[val, 0] - 1) * 2)
        else:
            da3cost.append(temp_matrix[val, 0] * 2)


def update_text_files():
    with open(prefix + 'percentcorrect.txt', 'w') as f:
        f.write(str(percent_correct))
    np.savetxt(prefix + "l1weights.txt", l1.weights)
    np.savetxt(prefix + "l2weights.txt", l2.weights)
    np.savetxt(prefix + "l3weights.txt", l3.weights)
    np.savetxt(prefix + "l1biases.txt", l1.biases)
    np.savetxt(prefix + "l2biases.txt", l2.biases)
    np.savetxt(prefix + "l3biases.txt", l3.biases)


def get_weights(layer_num):
    if layer_num == 1:
        return np.loadtxt(prefix + "l1weights.txt")
    elif layer_num == 2:
        return np.loadtxt(prefix + "l2weights.txt")
    elif layer_num == 3:
        return np.loadtxt(prefix + "l3weights.txt")


def get_biases(layer_num):
    if layer_num == 1:
        return np.loadtxt(prefix + "l1biases.txt")
    elif layer_num == 2:
        return np.loadtxt(prefix + "l2biases.txt")
    elif layer_num == 3:
        return np.loadtxt(prefix + "l3biases.txt")


# l1 = la.Layer(200)
# l2 = la.Layer(200, l1)
# l3 = la.Layer(47, l2)
l1 = la.Layer(100, weights=get_weights(1), biases=np.rot90([get_biases(1)], 3))
l2 = la.Layer(100, l1, weights=get_weights(2), biases=np.rot90([get_biases(2)], 3))
l3 = la.Layer(47, l2, weights=get_weights(3), biases=np.rot90([get_biases(3)], 3))


prop_c = 2.0


for j in range(300):
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
        total += mse((j % set_num) * set_size + i)

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
    if j % 1 == 0:
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

for i in range(test_size):
    l1.update(test_vect_arr[i // set_size][i % set_size])
    l2.update()
    l3.update()
    maxim = 0
    maxim_index = 0
    for j in range(len(l3.matrix)):
        if l3.matrix[j, 0] > maxim:
            maxim = l3.matrix[j, 0]
            maxim_index = j
    print(f'{maxim_index} {test_labels[i]}')
    # print(test_labels[i])
    if maxim_index == test_labels[i]:
        correct += 1
    else:
        print("!!!")
    tot += 1
percent_correct = correct/tot
if percent_correct > float(np.loadtxt(prefix + "percentcorrect.txt")) and tot == test_size:
    update_text_files()

print(f'{correct} / {tot} - {percent_correct}')
