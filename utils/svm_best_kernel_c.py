import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def get_svm_confusion_matrix(cow_df: pd.DataFrame, grass_df: pd.DataFrame, sky_df: pd.DataFrame):
    output_dir = 'output/graphs/'
    os.makedirs(output_dir, exist_ok=True)

    iters = 10
    kernels = ['linear', 'sigmoid', 'rbf', 'poly']
    # Define the range of C values:
    # - From 0.5 to 5 in steps of 0.5
    # - From 6 to 20 in steps of 1
    # c_values = np.concatenate((np.arange(0.5, 2.5, 0.5), np.arange(3, 21, 1)))
    c_values = [0.5, 1, 2.5, 5, 10, 25, 50]

    df = pd.concat([cow_df, grass_df, sky_df], ignore_index=True)
    dataframes_list = []

    for _ in range(iters):
        train_df, test_df = train_test_split(df, test_size=0.2)  # Adjust test_size as needed
        x_train = train_df[["R", "G", "B"]].to_numpy()
        y_train = train_df["Class"].to_numpy()
        dataframes_list.append((train_df, test_df, x_train, y_train))

    class_labels = np.array(['cow', 'grass', 'sky'])

    for kernel in kernels:
        for c in c_values:
            to_predict = []
            predictions = []

            for index in range(iters):
                train_df, test_df, x_train, y_train = dataframes_list[index]
                x_test = test_df[["R", "G", "B"]].to_numpy()
                y_test = test_df["Class"].to_numpy()

                svm = SVC(C=c, kernel=kernel)
                svm.fit(x_train, y_train)

                iteration_predictions = svm.predict(x_test)
                to_predict.extend(y_test)
                predictions.extend(iteration_predictions)

            cm = confusion_matrix(to_predict, predictions, labels=class_labels, normalize='true')

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=sns.cubehelix_palette(as_cmap=True, rot=.2, gamma=.5))
            plt.title(f"Normalized Confusion Matrix for {kernel} kernel, c = {c:.1f}")
            plt.colorbar()

            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels, rotation=45)
            plt.yticks(tick_marks, class_labels)

            fmt = '.2f'
            thresh = cm.max() / 2.
            for i in range(len(class_labels)):
                for j in range(len(class_labels)):
                    plt.text(j, i, format(cm[i, j], fmt),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            os.makedirs(f"output/graphs/{kernel}", exist_ok=True)
            plt.savefig(f"output/graphs/{kernel}/confusion_matrix_{c:.1f}.png", bbox_inches='tight', dpi=1200)
            plt.close()

            # Calculate precision and accuracy
            precision = precision_score(to_predict, predictions, average='weighted')
            accuracy = accuracy_score(to_predict, predictions)
            error_precision = 1 - precision
            error_accuracy = 1 - accuracy

            print(f"{kernel} kernel, c = {c:.1f}")
            print(f"Precision: {precision:.5f}, Error (Precision): {error_precision:.5f}")
            print(f"Accuracy: {accuracy:.5f}, Error (Accuracy): {error_accuracy:.5f}")
