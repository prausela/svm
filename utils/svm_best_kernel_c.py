import numpy as np
import pandas as pd

from utils.confusion_matrix import calculate_relative_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.data_split import k_fold_split
from utils.plotter import plot_confusion_matrix
from utils.svm_utils import get_svm_by_c_kernel


def print_results(conf_matrix, per_label_conf_matrix):
    print("Confusion Matrix:")
    print(conf_matrix)

    # print("Per Label Confusion Matrix:")
    # print(per_label_conf_matrix)

    metrics_result = metrics(per_label_conf_matrix)

    precision_values = [values['Precision'] for values in metrics_result.values()]
    accuracy_values = [values['Accuracy'] for values in metrics_result.values()]

    average_precision = sum(precision_values) / len(metrics_result)
    average_accuracy = sum(accuracy_values) / len(metrics_result)

    precision_std = np.std(precision_values)
    accuracy_std = np.std(accuracy_values)

    print(f'Average Precision: {average_precision}')
    print(f'Average Accuracy: {average_accuracy}')
    print(f'Standard Deviation of Precision: {precision_std}')
    print(f'Standard Deviation of Accuracy: {accuracy_std}')


def get_svm_confusion_matrix(cow_df: pd.DataFrame, grass_df: pd.DataFrame, sky_df: pd.DataFrame):
    iters = 1
    kernels = ['linear', 'sigmoid', 'rbf', 'poly']
    c_values = np.arange(0.2, 2.2, 0.2)  # Start from 0.2, end at 2.0, with a step of 0.2
    df = pd.concat([cow_df, grass_df, sky_df], ignore_index=True)

    dataframes_list = []
    for _ in range(iters):
        train_df, test_df = k_fold_split(df, k=5)
        x = train_df[["R", "G", "B"]].to_numpy()
        y = train_df["Class"].to_numpy()
        dataframes_list.append((train_df, test_df, x, y))

    class_labels = np.array(['cow', 'grass', 'sky'])
    for kernel in kernels:
        for c in c_values:
            print(f"{kernel} {c}")
            predictions = []
            to_predict = []
            for index in range(iters):
                train_df, test_df, x, y = dataframes_list[index]
                svm = get_svm_by_c_kernel(x, y, c, kernel)
                print(f"{kernel} {c} svm generated")
                predictions.append(svm.predict(test_df[["R", "G", "B"]]))
                to_predict.append(test_df["Class"])
                print(f"{kernel} {c} values predicted")
            result_df = pd.DataFrame({'predictions': predictions, 'to_predict': to_predict})
            print(result_df)
            conf_mat = calculate_relative_confusion_matrix(class_labels, result_df['predictions'],
                                                           result_df['to_predict'])
            per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_mat)
            print_results(conf_mat, per_label_conf_matrix)
            plot_confusion_matrix(conf_mat, "Matriz de confusión", f"./output/graphics/{kernel}{c}_conf_mat.png", ".2f")
