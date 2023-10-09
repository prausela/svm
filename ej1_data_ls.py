import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_ls_data(num_points, separator, x_range=(0, 5), y_range=(0, 5), filename="TP3-1.csv"):

    data = []
    
    for _ in range(num_points):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])

        if y >= separator(x):
            label = 1  
        else:
            label = -1 
        
        data.append((x, y, label))

    df = pd.DataFrame(data, columns=["X", "Y", "Label"])
    df.to_csv(filename)
    
    return data

def generate_ls_data_mc(num_points, num_misclassified, separator, x_range=(0, 5), y_range=(0, 5), filename="TP3-2.csv"):

    data = []
    
    for _ in range(num_points - num_misclassified):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])

        if y >= separator(x):
            label = 1
        else:
            label = -1

        data.append((x, y, label))
    
    for _ in range(num_misclassified):
        x = np.random.uniform(x_range[0], x_range[1])
        y = separator(x) + np.random.uniform(-0.5, 0.5)

        if y >= separator(x):
            label = -1
        else:
            label = 1

        data.append((x, y, label))

    df = pd.DataFrame(data, columns=["X", "Y", "Label"])
    df.to_csv(filename)

    return data

num_points = 100
separator = lambda x: 3 * x - 2 
data_ls = generate_ls_data(num_points , separator)
    

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
