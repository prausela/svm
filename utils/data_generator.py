import numpy as np
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
    df.to_csv(filename, index=False)
    
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
    df.to_csv(filename, index=False)

    return data
