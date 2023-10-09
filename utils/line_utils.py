from utils.perceptron import Perceptron

import pandas as pd
import numpy as np

class Line:

    def __init__(self, w: np.ndarray) -> None:
        pass

    def evaluate(self, point: np.ndarray) -> float:
        pass

    def coefficients(self) -> np.ndarray:
        pass

def step_perceptron_line(perceptron: Perceptron) -> Line:
    pass

def sample2points(df: pd.DataFrame, out_col: str, perceptron: Perceptron = None, 
                  only_correct: bool = False) -> tuple[np.ndarray, np.ndarray]:
    pass

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

