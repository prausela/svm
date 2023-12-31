from heapq import heappush, heappushpop
from utils.activation_functions import step_activation
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
        coeffs = single_row_array_shape(self.coeffs)
        return np.copy(coeffs)

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

    if class_pick_count > class_count:
        raise ValueError("Must provide number less or equal to {} for {} points. "
                         "{} were provided".format(class_count, class_name, class_pick_count))
    
    class_distances = distances[class_idxs]
    class_distances = abs(class_distances)

    class_dist_idxs = least_dist_idxs(class_distances, class_pick_count)
    return class_idxs[class_dist_idxs]


def __least_dist2line_points_idxs_by_class_for_each__(points: np.ndarray, line: Line,
                                                      positive_points: int, 
                                                      negative_points: int
                                                      ) -> tuple[np.ndarray, np.ndarray]:
    point_count = positive_points + negative_points

    n_rows = points.shape[0]
    if point_count > n_rows:
        raise ValueError("Must provide number of points less or equal to {}. "
                         "{} were provided: {} positive and {} negative".format(n_rows, point_count, 
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
    if point_count > n_rows:
        raise ValueError("Must provide number of points less or equal to {}. "
                         "{} were provided".format(n_rows, point_count))
    
    distances = np.apply_along_axis(line.distance_to, 1, points)
    abs_dists = abs(distances)

    dist_idxs = least_dist_idxs(abs_dists, point_count)
    
    selected_dists = distances[dist_idxs]

    selected_dists_pos_cond = selected_dists >= 0
    selected_dists_neg_cond = selected_dists < 0

    some_positive = np.any(selected_dists_pos_cond)
    some_negative = np.any(selected_dists_neg_cond)
    if some_positive and some_negative:
        return dist_idxs[selected_dists_pos_cond], dist_idxs[selected_dists_neg_cond]
    
    if not some_positive:
        other_class_dist_idxs = __least_dist_points_idxs_by_class_for_class__(distances, distances >= 0, 1, "positive")
    else:
        other_class_dist_idxs = __least_dist_points_idxs_by_class_for_class__(distances, distances < 0, 1, "negative")

    dist_idxs[0] = other_class_dist_idxs[0]

    return dist_idxs[selected_dists_pos_cond], dist_idxs[selected_dists_neg_cond]


def least_dist2line_points_idxs_by_class(points: np.ndarray, line: Line, point_count: int = None,
                                         positive_points: int = None, negative_points: int = None
                                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    point_count_is_none = point_count is None
    positive_and_negative_points_are_none = positive_points is None and negative_points is None
    
    if point_count_is_none and positive_and_negative_points_are_none:
        raise ValueError("Must provide either point_count or positive_points and negative_points. "
                         "Currently, all are None")
    
    if not point_count_is_none and not positive_and_negative_points_are_none:
        raise ValueError("Must provide either point_count or positive_points and negative_points. "
                         "Not both")
    
    if not point_count_is_none and point_count < 3:
        raise ValueError("Must specify a number of points greater or equal to {}. "
                         "Specified {}".format(3, point_count))
    
    if positive_and_negative_points_are_none:
        return __least_dist2line_points_idxs_by_class_for_all__(points, line, point_count)
    
    if positive_points is None and negative_points is not None:
        positive_points = point_count - negative_points
    elif positive_points is not None and negative_points is None:
        negative_points = point_count - positive_points 

    point_count = positive_points + negative_points
    if point_count < 3:
        raise ValueError("Must specify a number of points greater or equal to {}. "
                         "Specified {}: {} positive {} negative".format(3, point_count, 
                                                                       positive_points, 
                                                                       negative_points))
    
    return __least_dist2line_points_idxs_by_class_for_each__(points, line, positive_points, negative_points)

def points_distances_to_line(line: Line, points: np.ndarray, class_sign: np.ndarray) -> np.ndarray:
    distances = np.apply_along_axis(line.distance_to, 1, points)
    distances = distances * class_sign
    return distances

X_POS = 0
Y_POS = 1

def support_vectors_line(dirPointA: np.ndarray, dirPointB: np.ndarray, traslPoint: np.ndarray) -> tuple[Line, Line]:
    x1, x2, x_t = dirPointA[X_POS], dirPointB[X_POS], traslPoint[X_POS]
    y1, y2, y_t = dirPointA[Y_POS], dirPointB[Y_POS], traslPoint[Y_POS]

    m =  (y2 - y1) / (x2 - x1)

    b_dir   = y2 - m * x2
    b_trasl = y_t - m * x_t

    min_b = min(b_dir, b_trasl)
    max_b = max(b_dir, b_trasl)

    b = min_b + (max_b - min_b) / 2
    
    wA = np.array([[m, -1, b]])
    wB = np.array([[-m, 1, -b]])

    lineA = Line(wA)
    lineB = Line(wB)
    return lineA, lineB


def __calculate_class_margin__(line: Line, class_points: np.ndarray, class_pred: np.ndarray) -> float:
    class_dists = points_distances_to_line(line, class_points, class_pred)
    class_least_dist_idx = least_dist_idxs(class_dists, 1).item()
    class_margin = class_dists[class_least_dist_idx]
    return class_margin
    
def __margin2line_from_points__(line: Line, classA_points: np.ndarray, classB_points: np.ndarray,
                                classA_pred: np.ndarray, classB_pred: np.ndarray) -> float:
    classA_margin = __calculate_class_margin__(line, classA_points, classA_pred)
    if classA_margin < 0:
        classA_margin = float("-inf")
    classB_margin = __calculate_class_margin__(line, classB_points, classB_pred)
    if classB_margin < 0:
        classB_margin = float("-inf")
    margin = classA_margin + classB_margin
    return margin

def max_margin_line_from_points(dir_points: np.ndarray, trasl_points: np.ndarray,
                                dir_pred: np.ndarray, trasl_pred: np.ndarray
                                ) -> tuple[float, Line, tuple[np.ndarray, np.ndarray]]:
    dir_points_count = dir_points.shape[0]
    trasl_points_count = trasl_points.shape[0]
    max_margin = None
    max_margin_line = None
    chosen_points_idxs = None
    for dir_i in range(dir_points_count):
        for dir_j in range(dir_i+1, dir_points_count):
            for trasl_idx in range(trasl_points_count):
                curr_points_idxs = (np.array([dir_i, dir_j]), np.array([trasl_idx]))
                lineA, lineB = support_vectors_line(dir_points[dir_i], dir_points[dir_j], trasl_points[trasl_idx])

                margin = __margin2line_from_points__(lineA, dir_points, trasl_points, dir_pred, trasl_pred)
                if max_margin is None or margin > max_margin:
                    max_margin = margin
                    max_margin_line = lineA
                    chosen_points_idxs = curr_points_idxs
                
                margin = __margin2line_from_points__(lineB, dir_points, trasl_points, dir_pred, trasl_pred)
                if max_margin is None or margin > max_margin:
                    max_margin = margin
                    max_margin_line = lineB
                    chosen_points_idxs = curr_points_idxs

    return max_margin, max_margin_line, chosen_points_idxs

def __max_margin_line_by_class_points__(positive_points: np.ndarray, negative_points: np.ndarray,
                                        pos_pred: np.ndarray, neg_pred: np.ndarray) -> tuple[float, Line,
                                                                                             tuple[np.ndarray, 
                                                                                                   np.ndarray]]:
    dir_by_pos_margin, dir_by_pos_line, dir_by_pos_chosen_idxs = max_margin_line_from_points(positive_points, negative_points, pos_pred, neg_pred)
    dir_by_neg_margin, dir_by_neg_line, dir_by_neg_chosen_idxs = max_margin_line_from_points(negative_points, positive_points, neg_pred, pos_pred)

    if dir_by_neg_margin is None or (dir_by_pos_margin is not None and dir_by_pos_margin >= dir_by_neg_margin):
        return dir_by_pos_margin, dir_by_pos_line, dir_by_pos_chosen_idxs

    return dir_by_neg_margin, dir_by_neg_line, (dir_by_neg_chosen_idxs[1], dir_by_neg_chosen_idxs[0])

def maximize_step_perceptron_line_margin(df: pd.DataFrame, out_col: str, perceptron: Perceptron, 
                                         points2decide : int = None, positive_points2decide: int = None, 
                                         negative_points2decide: int = None, only_correct : bool = True
                                         ) :#-> tuple[Perceptron, float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    
    points, _ = sample2points(df, out_col, perceptron, only_correct)
    line = step_perceptron_line(perceptron)
    pos_idxs, neg_idxs = least_dist2line_points_idxs_by_class(points, line, points2decide, 
                                                              positive_points2decide, 
                                                              negative_points2decide)
    sel_points_pos = points[pos_idxs]
    sel_points_neg = points[neg_idxs]
    pos_pred = np.apply_along_axis(lambda x : perceptron.predict(x=x), 1, sel_points_pos)
    neg_pred = np.apply_along_axis(lambda x : perceptron.predict(x=x), 1, sel_points_neg)
    margin, line, chosen_points_class_idxs = __max_margin_line_by_class_points__(sel_points_pos, sel_points_neg, pos_pred, neg_pred)
    max_margin_perceptron = Perceptron(line.coefficients(), step_activation)
    chosen_points_pos_class_idxs = chosen_points_class_idxs[0]
    chosen_points_neg_class_idxs = chosen_points_class_idxs[1]
    chosen_points_pos = points[pos_idxs[chosen_points_pos_class_idxs]]
    chosen_points_neg = points[neg_idxs[chosen_points_neg_class_idxs]]
    
    return max_margin_perceptron, margin, points, (chosen_points_pos, chosen_points_neg), (sel_points_pos, sel_points_neg)

