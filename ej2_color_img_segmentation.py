import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_split import k_fold_split
from utils.plot_utils import plot_predictions
from utils.svm_utils import get_svm_by_c_kernel
from utils.img_utils import img_to_classified_df, img_to_df, get_img_size


def run_ej2():

    ''' Ej 2:
        Segmentación de imágenes a color.
    '''
    
    # Ej 2.A
    cow_df = img_to_classified_df("data/vaca.jpg", 'cow')
    grass_df = img_to_classified_df("data/pasto.jpg", 'grass')
    sky_df = img_to_classified_df("data/cielo.jpg", 'sky')
    df = pd.concat([cow_df, grass_df, sky_df], ignore_index=True)

    # Ej 2.B
    train_df, test_df = k_fold_split(df, k=5)
    
    # Ej 2.C Part 1
    X = train_df[["R", "G", "B"]].to_numpy()
    Y = train_df["Class"].to_numpy()
    svm = get_svm_by_c_kernel(X, Y, c=1, kernel="rbf")

    # Ej 2.C Part 2
    # TODO: Cross-validation + conf matrix of diff combinations of Cs and kernels
    
    # Ej 2.D
    # TODO: Get best kernel
    
    # Ej 2.F
    sample_df = img_to_df("data/cow.jpg")
    sample_size = get_img_size("data/cow.jpg")
    predictions = svm.predict(sample_df)
    
    # Ej 2.E
    plot_predictions(predictions, sample_size)

    # Ej 2.G
    # TODO: Classify random img


if __name__ == '__main__':
    run_ej2()
