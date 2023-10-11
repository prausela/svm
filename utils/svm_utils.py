import numpy as np
import pandas as pd
from sklearn.svm import SVC
from typing import Tuple, Union

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


def l(t: Union[np.ndarray, float]):
    if t >= 1:
        return 0
    return 1 - t


def calculate_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, c: float) -> float:
    t = np.dot(y, (np.dot(x, w) + b))
    return 0.5 * np.dot(w, w) + c * np.sum(l(t))


def compute_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, c: float) -> Tuple[np.ndarray, float]:
    t = np.dot(y, (np.dot(x, w) + b))
    w_gradient = w
    b_gradient = 0
    if t < 1:
        w_gradient = w + c * np.sum(np.dot(y, x) * (-1))
        b_gradient = c * np.sum(y) * (-1)
    return w_gradient, b_gradient
