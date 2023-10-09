from utils.perceptron import Perceptron, __add_bias__, sample2points
from utils.matrix_utils import single_column_array_shape, single_row_array_shape, mat_norm_2

import pandas as pd
import numpy as np

class Line:

    def __init__(self, coeffs: np.ndarray) -> None:
        coeffs = single_column_array_shape(coeffs, 3)

        no_bias_coeffs = coeffs[:-1, :]
        self.coeffs = coeffs / mat_norm_2(no_bias_coeffs)

    def distance_to(self, point: np.ndarray) -> float:
        point = single_row_array_shape(point, 2)
        point = __add_bias__(point)

        single_val_mat = np.dot(point, self.coeffs)

        return single_val_mat.item()

    def coefficients(self) -> np.ndarray:
        return np.copy(self.coeffs)

def step_perceptron_line(perceptron: Perceptron) -> Line:
    return Line(perceptron.w)

def point_idxs_by_dist2line(points: np.ndarray, line: Line, point_count: int = None,
                            positive_points: int = None, negative_points: int = None) -> np.ndarray:
    pass

def max_margin_line_from_points(positive_points: np.ndarray, negative_points: np.ndarray) -> Line:
    pass

def support_vectors_line(dirPointA: np.ndarray, dirPointB: np.ndarray, traslPoint: np.ndarray) -> Line:
    pass

def maximize_step_perceptron_line_margin(df: pd.DataFrame, out_col: str, perceptron: Perceptron, 
                                         points2decide : int = None, positive_points: int = None, negative_points: int = None,
                                         only_correct : bool = True) -> Perceptron:
    pass

