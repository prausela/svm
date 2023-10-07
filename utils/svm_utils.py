import pandas as pd
from sklearn.svm import SVC

from utils.img_utils import img_to_df, get_img_size
from utils.plot_utils import plot_predictions
from utils.sklearn_utils import c_to_sklearn_c


def get_svm_by_c_kernel(x: pd.DataFrame, y: pd.DataFrame, c: int, kernel: str) -> SVC:
    sk_c = c_to_sklearn_c(c)
    svm = SVC(C=sk_c, kernel=kernel)
    svm.fit(x, y)
    return svm


def predict_and_plot(image_path: str, output_filename: str, svm: SVC):
    sample_df = img_to_df(image_path)
    sample_size = get_img_size(image_path)
    predictions = svm.predict(sample_df)

    plot_predictions(predictions, sample_size, output_filename)
