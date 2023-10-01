import numpy as np
import pandas as pd
from PIL import Image

def img_to_array(filename: str) -> np.ndarray:
    img = Image.open(filename)
    img_array = np.array(img.getdata())
    return img_array

def img_to_df(filename: str) -> pd.DataFrame:
    img_array = img_to_array(filename)
    df = pd.DataFrame(columns=['R', 'G', 'B'], data=img_array)
    return df

def img_to_classified_df(filename: str, label: str) -> pd.DataFrame:
    df = img_to_df(filename)
    df = df.assign(Class=label)
    return df