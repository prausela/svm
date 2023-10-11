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
    def __init__(self, w: np.ndarray, activation_func_single: Callable[[float], float], 
                 perceptron_type: str = None) -> None:
        self.w = w
        self.__activation_func_single__ = activation_func_single
        if perceptron_type is not None:
            self.type = perceptron_type
        elif activation_func_single in activation_function_types:
            self.type = activation_function_types[activation_func_single]
        else:
            self.type = "custom"

    def predict(self, df: pd.Series = None, x: np.ndarray = None) -> Any:
        if (df is None and x is None) or (df is not None and x is not None):
            raise ValueError("Must provide either a Pandas Series or a numpy array")
        
        if x is None:
            x = df.to_numpy()
        x = __column_mat__(x)
        activation_func = np.vectorize(self.__activation_func_single__)
        x = __add_bias__(x)
        O = __predict__(x, self.w, activation_func)
        return O.item()
    

def sample2points(df: pd.DataFrame, out_col: str, perceptron: Perceptron = None, 
                  only_correct: bool = None) -> tuple[np.ndarray, np.ndarray]:
    
    x_df = df.drop([out_col], axis="columns", inplace=False)
    y_df = df[[out_col]]

    if (only_correct is None and perceptron is None) or (only_correct is not None and not only_correct):

        x = x_df.to_numpy()
        y = y_df.to_numpy()

        return (x, y)
    
    if perceptron is None:
        raise ValueError("Must provide Perceptron when only_correct is True")
    
    O_df = x_df.apply(perceptron.predict, axis=1)

    only_correct_selection = y_df[out_col] == O_df

    x_df = x_df[only_correct_selection]
    y_df = y_df[only_correct_selection]

    x = x_df.to_numpy()
    y = y_df.to_numpy()

    return (x, y)


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

def build_perceptron(train_df: pd.DataFrame, test_df : pd.DataFrame, out_col : str, eta : float, 
                   activation_func_single: Callable[[float], float],
                   x: np.ndarray = None, y: np.ndarray = None,
                   iters: int = None,
                   calculate_error: Callable[[np.ndarray, np.ndarray, int], float] = None,
                   init_weights: Callable[[int], np.ndarray] = None,
                   random_state: np.random.Generator = None,
                   perceptron_type: str = None
                ) -> tuple[Perceptron, list[float], list[float], np.ndarray]:
    
    if not (((train_df is None and out_col is None) and (x is not None and y is not None)) or \
            ((train_df is not None and out_col is not None) and (x is None and y is None))):
        raise ValueError("Must specify either train_df and out_col or x and y")
    
    if __is_lambda__(activation_func_single):
        warnings.warn("Avoid using lambdas as activation functions as they are locally-bound")

    if random_state is None:
        random_state = np.random.default_rng()

    if train_df is not None:
        x, y = sample2points(train_df, out_col)

    x_test, y_test = sample2points(test_df, out_col)

    activation_func = np.vectorize(activation_func_single)
    prediction_func = lambda x, w: __predict__(x, w, activation_func)

    if init_weights is None:
        init_weights = init_random_between_vals_builder(-1, 1, random_state)

    if calculate_error is None:
        calculate_error = mse

    x = __add_bias__(x)
    x_test = __add_bias__(x_test)

    p = x.shape[0]
    n = x.shape[1]

    p_test = x_test.shape[0]

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
    test_error_per_iter = []

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

        O_test = prediction_func(x_test, w)
        test_error = calculate_error(y_test, O_test, p_test)
        test_error_per_iter.append(test_error)

        if min_error is None or error < min_error:
            min_error = error
            w_min = w
        i += 1

    return (Perceptron(w_min, activation_func_single, perceptron_type), error_per_iter, test_error_per_iter, init_w)


def build_step_perceptron(train_df: pd.DataFrame, test_df : pd.DataFrame, out_col: str, eta: float,
                          x: np.ndarray = None, y: np.ndarray = None,
                          iters: int = None,
                          calculate_error: Callable[[np.ndarray, np.ndarray, np.ndarray, int], float] = None,
                          init_weights: Callable[[int], np.ndarray] = None,
                          random_state: np.random.Generator = None
                          ) -> tuple[Perceptron, list[float], list[float], np.ndarray]:
    return build_perceptron(train_df, test_df, out_col, eta, step_activation, x, y, 
                            iters, calculate_error, init_weights, random_state)
