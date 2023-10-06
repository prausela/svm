import pandas as pd
from sklearn.svm import SVC
from utils.sklearn_utils import c_to_sklearn_c

def get_svm_by_c_kernel(X: pd.DataFrame, Y: pd.DataFrame, c: int, kernel: str) -> SVC:
    sk_c = c_to_sklearn_c(c)
    svm = SVC(C=sk_c, kernel=kernel)
    svm.fit(X, Y)
    return svm