import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_generator import generate_ls_data, generate_ls_data_mc


num_points = 100
separator = lambda x: 3 * x - 2
data_ls = generate_ls_data(num_points, separator)

class_A = [(x, y) for x, y, label in data_ls if label == 1]
class_B = [(x, y) for x, y, label in data_ls if label == -1]

class_A_x, class_A_y = zip(*class_A)
class_B_x, class_B_y = zip(*class_B)

plt.figure(figsize=(8, 6))
plt.scatter(class_A_x, class_A_y, c='b', marker='o', label='Class A')
plt.scatter(class_B_x, class_B_y, c='r', marker='x', label='Class B')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linearly Separable Data')
plt.show()

num_misclassified = 20
data_misclassification = generate_ls_data_mc(num_points, num_misclassified, separator)

class_A = [(x, y) for x, y, label in data_misclassification if label == 1]
class_B = [(x, y) for x, y, label in data_misclassification if label == -1]
class_A_x, class_A_y = zip(*class_A)
class_B_x, class_B_y = zip(*class_B)
plt.figure(figsize=(8, 6))
plt.scatter(class_A_x, class_A_y, c='b', marker='o', label='Class 1')
plt.scatter(class_B_x, class_B_y, c='r', marker='x', label='Class -1')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linearly Separable Data with Misclassification')
plt.show()
