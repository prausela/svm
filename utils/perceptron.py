from typing import Callable, Any
from utils.error_functions import mse
from utils.activation_functions import step_activation, activation_function_types
from utils.init_functions import init_random_between_vals_builder

import numpy as np
import pandas as pd
import warnings


def __column_mat__(x: np.ndarray) -> np.ndarray:
    return x.reshape((1, -1))


class Perceptron:
    def __init__(self, w : np.ndarray, activation_func_single : Callable[[float], float], perceptron_type : str = None) -> None:
        self.w = w
        self.__activation_func_single__ = activation_func_single
        if perceptron_type is not None:
            self.type = perceptron_type
        elif activation_func_single in activation_function_types:
            self.type = activation_function_types[activation_func_single]
        else:
            self.type = "custom"

    def predict(self, df: pd.Series) -> Any:
        x = df.to_numpy()
        x = __column_mat__(x)
        activation_func = np.vectorize(self.__activation_func_single__)
        x = __add_bias__(x)
        O = __predict__(x, self.w, activation_func)
        return O.item()


def __add_bias__(x: np.ndarray) -> np.ndarray:
    p = x.shape[0]
    n = x.shape[1]

    aux = np.ones((p, n + 1))
    aux[:, :n] = x

    return aux


def __predict__(x: np.ndarray, w: np.ndarray, activation_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    h = np.dot(x, w.transpose())
    O = activation_func(h)
    return O


def __is_lambda__(f: Callable) -> bool:
    return f.__code__.co_name == "<lambda>"

def build_perceptron(train_df : pd.DataFrame, out_col : str, eta : float, 
                   activation_func_single : Callable[[float], float],
                   iters : int = None,
                   calculate_error : Callable[[np.ndarray, np.ndarray, int], float] = None,
                   init_weights : Callable[[int], np.ndarray] = None,
                   random_state : np.random.Generator = None,
                   perceptron_type : str = None
                ) -> tuple[Perceptron, list[float]]:
    
    if __is_lambda__(activation_func_single):
        warnings.warn("Avoid using lambdas as activation functions as they are locally-bound")

    if random_state is None:
        random_state = np.random.default_rng()

    x = train_df.drop([out_col], axis="columns", inplace=False).to_numpy()
    y = train_df[[out_col]].to_numpy()

    activation_func = np.vectorize(activation_func_single)
    prediction_func = lambda x, w: __predict__(x, w, activation_func)

    if init_weights is None:
        init_weights = init_random_between_vals_builder(-1, 1, random_state)

    if calculate_error is None:
        calculate_error = mse

    x = __add_bias__(x)

    p = x.shape[0]
    n = x.shape[1]

    i = 0
    error = None
    min_error = p * 2

    w = init_w = init_weights(n)

    if not isinstance(w, np.ndarray):
        raise ValueError("Weight initialization failed. Weights should be a numpy array")

    if w.shape != (1, n):
        raise ValueError("Weight initialization failed. Should be shape (1, {})".format(n))

    w_min = None

    error_per_iter = []

    while (error is None or error > 0) and (iters is None or i < iters):

        i_idx = random_state.integers(0, p)
        i_idx_next = i_idx + 1

        x_i = x[i_idx:i_idx_next]
        y_i = y[i_idx:i_idx_next]

        O_i = prediction_func(x_i, w)

        deltaW = eta * (y_i - O_i) * x_i
        w = w + deltaW

        O = prediction_func(x, w)
        error = calculate_error(y, O, p)
        error_per_iter.append(error)
        if min_error is None or error < min_error:
            min_error = error
            w_min = w
        i += 1

    return (Perceptron(w_min, activation_func_single, perceptron_type), error_per_iter, init_w)


def build_step_perceptron(train_df: pd.DataFrame, out_col: str, eta: float,
                          iters: int = None,
                          calculate_error: Callable[[np.ndarray, np.ndarray, np.ndarray, int], float] = None,
                          init_weights: Callable[[int], np.ndarray] = None,
                          random_state: np.random.Generator = None
                          ) -> tuple[Perceptron, list[float]]:
    return build_perceptron(train_df, out_col, eta, step_activation, iters, calculate_error, init_weights, random_state)
