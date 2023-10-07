import pandas as pd

from utils.data_split import k_fold_split
from utils.svm_utils import get_svm_by_c_kernel, predict_and_plot
from utils.img_utils import img_to_classified_df


def run_ej2():
    """ Ej 2:
        Segmentación de imágenes a color.
    """

    # Ej 2.A
    cow_df = img_to_classified_df("data/vaca.jpg", 'cow')
    grass_df = img_to_classified_df("data/pasto.jpg", 'grass')
    sky_df = img_to_classified_df("data/cielo.jpg", 'sky')
    df = pd.concat([cow_df, grass_df, sky_df], ignore_index=True)

    # Ej 2.B
    train_df, test_df = k_fold_split(df, k=5)

    # Ej 2.C Part 1
    x = train_df[["R", "G", "B"]].to_numpy()
    y = train_df["Class"].to_numpy()
    svm = get_svm_by_c_kernel(x, y, c=1, kernel="rbf")

    # Ej 2.C Part 2
    # TODO: Cross-validation + conf matrix of diff combinations of Cs and kernels

    # Ej 2.D
    # TODO: Get best kernel

    # Ej 2.F, 2.E predict + plot
    predict_and_plot("data/cow.jpg", "classified_cow.png", svm)

    # Ej 2.G
    predict_and_plot("data/milka-cow.jpg", "classified_milka_cow.png", svm)


if __name__ == '__main__':
    run_ej2()
