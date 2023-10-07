from typing                     import Callable
from utils.error_functions      import mse
from utils.activation_functions import step_activation

import numpy  as np
import pandas as pd
import random

class Perceptron:
    def __init__(self, w : np.ndarray, activation_func_single : Callable[[float], float]) -> None:
        self.w = w
        self.activation_func_single = activation_func_single

    def predict(self, x : np.ndarray) -> np.ndarray:
        activation_func = np.vectorize(self.activation_func_single)
        x = __add_bias__(x)
        return __predict__(x, self.w, activation_func)

def __add_bias__(x : np.ndarray) -> np.ndarray:
    p = x.shape[0]
    n = x.shape[1]

    aux = np.ones((p, n+1))
    aux[:, :n] = x

    return aux

def __predict__(x : np.ndarray, w : np.ndarray, activation_func : Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    h = np.dot(x, w.transpose())
    O = activation_func(h)
    return O

def build_perceptron(train_df : pd.DataFrame, out_col : str, eta : float, 
                   activation_func_single : Callable[[float], float],
                   iters : int,
                   calculate_error : Callable[[np.ndarray, np.ndarray, np.ndarray, int], float] = None
                ) -> tuple[Perceptron, list[float]]:
    
    x = train_df.drop([out_col], axis="columns", inplace=False).to_numpy()
    y = train_df[[out_col]].to_numpy()

    activation_func = np.vectorize(activation_func_single)
    prediction_func = lambda x, w : __predict__(x, w, activation_func)

    if calculate_error is None:
        calculate_error = lambda _, y, O, p : mse(y, O, p)

    x = __add_bias__(x)
    
    p = x.shape[0]
    n = x.shape[1]

    i     = 0
    error = None
    min_error = p * 2

    w = np.random.random((1, n)) * 2 - 1
    w_min = None

    error_per_iter = []

    while (error is None or error > 0) and i < iters:
        
        x_idx = random.randint(0, p-1)
        x_i = x[x_idx:x_idx+1]
        y_i = y[x_idx:x_idx+1]
        
        O_i = prediction_func(x_i, w)
        
        deltaW = eta * (y_i - O_i) * x_i
        w = w + deltaW
        
        O = prediction_func(x, w)
        error = calculate_error(x, y, O, p)
        error_per_iter.append(error)
        if min_error is None or error < min_error:
            min_error = error
            w_min = w
        i += 1

    return (Perceptron(w_min, activation_func_single), error_per_iter)

def build_step_perceptron(train_df : pd.DataFrame, out_col : str, eta : float, 
                   iters : int,
                   calculate_error : Callable[[np.ndarray, np.ndarray, np.ndarray, int], float] = None
                ) -> tuple[Perceptron, list[float]]:
    return build_perceptron(train_df, out_col, eta, step_activation, iters, calculate_error)