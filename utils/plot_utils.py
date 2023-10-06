import numpy as np
import matplotlib.pyplot as plt
from utils.img_utils import class_colors

def plot_predictions(predictions: np.ndarray, sample_size: dict):
    predictions_color = np.array([[class_colors[pred]] for pred in predictions])
    predictions_color_matrix = predictions_color.reshape(sample_size['height'], sample_size['width'], 3)
    fig, ax = plt.subplots()
    ax.imshow(predictions_color_matrix)
    plt.savefig('output/classified_cow.png')