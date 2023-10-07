import numpy as np

def mse(y : np.ndarray, O : np.ndarray, p : int) -> float:
    err = (y - O)

    sum_sqr_err_mat = np.dot(err.transpose(), err)
    sum_sqr_err = sum_sqr_err_mat.sum()

    return sum_sqr_err / p