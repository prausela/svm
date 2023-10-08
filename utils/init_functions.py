from numpy.typing import ArrayLike
from typing       import Callable

import numpy as np

def __init_random_between_vals__(n : int, lower_limit : float, upper_limit : float, random_state : np.random.Generator = None) -> np.ndarray:
    if random_state is None:
        random_state = np.random.default_rng()

    if lower_limit > upper_limit:
        raise 'Lower limit cannot be greater than upper limit'
    
    limits_range = upper_limit - lower_limit
    return random_state.random(size=(1, n)) * limits_range + lower_limit

def __init_random_sample_vals__(n : int, to_sample : ArrayLike, random_state : np.random.Generator = None) -> np.ndarray:
    if random_state is None:
        random_state = np.random.default_rng()

    return random_state.choice(to_sample, size=(1, n), replace=True)

def __init_same_val__(n : int, value : float) -> np.ndarray:
    return np.ones((1, n)) * value

def init_random_between_vals_builder(lower_limit : float, upper_limit : float, random_state : np.random.Generator = None) -> Callable[[int], np.ndarray]:

    def init_random_between_vals(n : int) -> np.ndarray:
        return __init_random_between_vals__(n, lower_limit, upper_limit, random_state)
    
    return init_random_between_vals

def init_random_sample_vals_builder(to_sample : ArrayLike, random_state : np.random.Generator = None) -> Callable[[int], np.ndarray]:

    def init_random_sample_vals(n : int) -> np.ndarray:
        return __init_random_sample_vals__(n, to_sample, random_state)
    
    return init_random_sample_vals

def init_same_val_builder(value : float, random_state : np.random.Generator = None) -> Callable[[int], np.ndarray]:

    def init_same_val(n : int) -> np.ndarray:
        return __init_same_val__(n, value, random_state)
    
    return init_same_val