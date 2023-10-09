import numpy as np
import math

def single_row_array_shape(a: np.ndarray, expected_size: int = None, 
                                err_msg: str = None) -> np.ndarray:
    return single_axis_array_shape(a, 0, expected_size, err_msg)

def single_column_array_shape(a: np.ndarray, expected_size: int = None, 
                                err_msg: str = None) -> np.ndarray:
    return single_axis_array_shape(a, 1, expected_size, err_msg)

def single_axis_array_shape(a: np.ndarray, axis: int, expected_size: int = None, 
                                err_msg: str = None) -> np.ndarray:
    if expected_size is None:
        expected_size = a.size

    if err_msg is None:
        err_msg = "Expected numpy array of size {}".format(expected_size)

    if not isinstance(a, np.ndarray):
            raise ValueError(err_msg)
    
    if a.size != expected_size:
        raise ValueError(err_msg)
    
    expected_shape = (1, expected_size) if axis == 0 \
                else (expected_size, 1)
        
    axis_size = a.shape[axis]
    if axis_size != 1:
        a = np.reshape(a, expected_shape)
    
    return a

def mat_norm_2(m: np.ndarray) -> np.ndarray:
    n_rows = m.shape[0]
    n_cols = m.shape[1]

    m_sqr = None
    if n_rows >= n_cols:
        m_sqr = np.dot(m.transpose(), m)
    else:
        m_sqr = np.dot(m, m.transpose())

    m_sqr_sum = np.sum(m_sqr)
    norm_2 = math.sqrt(m_sqr_sum)

    return norm_2