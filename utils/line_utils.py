from heapq import heappush, heappushpop
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

def least_dist_idxs(distances: np.ndarray, point_count: int) -> np.ndarray:
    least_distances_heap = []
    biggest_least_distance = None

    for dist_idx in range(distances.size):
        curr_dist = distances[dist_idx]

        if biggest_least_distance is None or curr_dist < biggest_least_distance:
            curr_dist_tuple = (curr_dist * -1, dist_idx)

            if len(least_distances_heap) >= point_count:
                heappushpop(least_distances_heap, curr_dist_tuple)

                neg_biggest_least_distance, _ = least_distances_heap[0]
                biggest_least_distance = neg_biggest_least_distance * -1
            else:
                heappush(least_distances_heap, curr_dist_tuple)

    dist_idxs = np.array([dist_tuple[1] for dist_tuple in least_distances_heap])
    return dist_idxs

def __least_dist_points_idxs_by_class_for_class__(distances: np.ndarray, class_condition: np.ndarray, 
                                                  class_pick_count: int, class_name: str) -> np.ndarray:
    class_tuple = np.where(class_condition)
    class_idxs = class_tuple[0]
    class_count = len(class_idxs)

    if class_count < class_pick_count:
        raise ValueError("Must provide number less or equal to {} for {} points. \
                         {} were provided".format(class_count, class_name, class_pick_count))
    
    class_distances = distances[class_tuple[0]]
    class_distances = abs(class_distances)

    class_dist_idxs = least_dist_idxs(class_distances, class_pick_count)
    return class_dist_idxs


def __least_dist2line_points_idxs_by_class_for_each__(points: np.ndarray, line: Line,
                                                      positive_points: int, 
                                                      negative_points: int
                                                      ) -> tuple[np.ndarray, np.ndarray]:
    point_count = positive_points + negative_points

    n_rows = points.shape[0]
    if n_rows < point_count:
        raise ValueError("Must provide number of points less or equal to {}. \
                         {} were provided: {} positive and {} negative".format(n_rows, point_count, 
                                                                               positive_points, negative_points))
    
    distances = np.apply_along_axis(line.distance_to, 1, points)

    positive_condition = distances >= 0
    negative_condition = np.invert(positive_condition)
    
    positive_dist_idxs = __least_dist_points_idxs_by_class_for_class__(distances, positive_condition, 
                                                                       positive_points, "positive")
    
    negative_dist_idxs = __least_dist_points_idxs_by_class_for_class__(distances, negative_condition, 
                                                                       negative_points, "negative")
    
    return (positive_dist_idxs, negative_dist_idxs)

def __least_dist2line_points_idxs_by_class_for_all__(points: np.ndarray, line: Line, point_count: int = None,
                                                     ) -> tuple[np.ndarray, np.ndarray]:

    n_rows = points.shape[0]
    if n_rows < point_count:
        raise ValueError("Must provide number of points less or equal to {}. \
                         {} were provided".format(n_rows, point_count))
    
    distances = np.apply_along_axis(line.distance_to, 1, points)
    abs_dists = abs(distances)

    dist_idxs = least_dist_idxs(abs_dists, point_count)
    
    selected_dists = distances[dist_idxs]

    some_positive = np.any(selected_dists >= 0)
    some_negative = np.any(selected_dists < 0)
    if some_positive and some_negative:
        return dist_idxs
    
    if not some_positive:
        other_class_dist_idxs = __least_dist_points_idxs_by_class_for_class__(distances, distances >= 0, 1, "positive")
    else:
        other_class_dist_idxs = __least_dist_points_idxs_by_class_for_class__(distances, distances < 0, 1, "negative")

    dist_idxs[0] = other_class_dist_idxs[0]

    return dist_idxs


def least_dist2line_points_idxs_by_class(points: np.ndarray, line: Line, point_count: int = None,
                                         positive_points: int = None, negative_points: int = None
                                         ) -> tuple[np.ndarray, np.ndarray]:
    
    point_count_is_none = point_count is None
    positive_and_negative_points_are_none = positive_points is None and negative_points is None
    
    if point_count_is_none and positive_and_negative_points_are_none:
        raise ValueError("Must provide either point_count or positive_points and negative_points. \
                         Currently, all are None")
    
    if not point_count_is_none and not positive_and_negative_points_are_none:
        raise ValueError("Must provide either point_count or positive_points and negative_points. \
                         Not both")
    
    if not point_count_is_none and point_count < 3:
        raise ValueError("Must specify a number of points greater than {}. \
                         Specified {}".format(3, point_count))
    
    if positive_and_negative_points_are_none:
        return __least_dist2line_points_idxs_by_class_for_all__(points, line, point_count)
    
    if positive_points is None and negative_points is not None:
        positive_points = point_count - negative_points
    elif positive_points is not None and negative_points is None:
        negative_points = point_count - positive_points 

    point_count = positive_points + negative_points
    if point_count < 3:
        raise ValueError("Must specify a number of points greater than {}. \
                         Specified {}: {} positive {} negative".format(3, point_count, 
                                                                       positive_points, 
                                                                       negative_points))
    
    return __least_dist2line_points_idxs_by_class_for_each__(points, line, positive_points, negative_points)


def max_margin_line_from_points(positive_points: np.ndarray, negative_points: np.ndarray) -> Line:
    pass

def support_vectors_line(dirPointA: np.ndarray, dirPointB: np.ndarray, traslPoint: np.ndarray) -> Line:
    pass

def maximize_step_perceptron_line_margin(df: pd.DataFrame, out_col: str, perceptron: Perceptron, 
                                         points2decide : int = None, positive_points: int = None, 
                                         negative_points: int = None, only_correct : bool = True
                                         ) -> Perceptron:
    pass

