import os
import sys

from utils.img_utils import img_to_classified_df
from utils.svm_best_kernel_c import get_svm_confusion_matrix, generate_matrix_csv


def generate_matrix_data(output_dir: str):
    cow_df = img_to_classified_df("data/vaca.jpg", 'cow')
    grass_df = img_to_classified_df("data/pasto.jpg", 'grass')
    sky_df = img_to_classified_df("data/cielo.jpg", 'sky')

    generate_matrix_csv(cow_df, grass_df, sky_df, output_dir, iters)


def get_best_values_ej2(input_dir: str):
    # Ej 2.C Part 2
    get_svm_confusion_matrix(float(c_value), kernel, input_dir, iters)

    # Ej 2.D
    # TODO: Get best kernel


if __name__ == '__main__':
    samples_dir = 'output/samples/'
    os.makedirs(samples_dir, exist_ok=True)
    iters = 10

    if len(os.listdir(samples_dir)) != iters * 2:
        generate_matrix_data(samples_dir)

    try:
        kernel = sys.argv[1]  # Trying to get first input from command line argument
        c_value = sys.argv[2]  # Trying to get second input from command line argument
    except IndexError:
        kernel = input("Please enter a kernel: ")
        c_value = input("Please enter a c_value: ")

    get_best_values_ej2(input_dir=samples_dir)
