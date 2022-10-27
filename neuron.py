# import random as r
# import math as m
#
#
# def sigmoid(val):
#     return 1 / (1 + m.exp(-val))
#
#
# class Neuron:
#     def __init__(self, connects):
#         self.value = 0
#         self.connects = connects
#         self.weights = []
#         for i in range(0, len(self.connects)):
#             self.weights.append(2*r.random() - 1)
#         self.bias = 2*r.random() - 1
#
#     def get_value(self):
#         return self.value
#
#     def get_weights(self):
#         return self.weights
#
#     def get_bias(self):
#         return self.bias
#
#     def update(self):
#         self.value = 0
#         for i in range(0, len(self.connects)):
#             for j in range(0, len(self.connects[0])):
#                 self.value += self.connects[i][j].get_value()*self.weights[i][j]
#         self.value += self.bias
#         self.value = sigmoid(self.value)
