import numpy as np
import pandas as pd
from PIL import Image

class_colors = {
    'cow': (255, 0, 0),  # Red
    'grass': (0, 255, 0),  # Green
    'sky': (0, 0, 255)  # Blue
}


def get_img_size(filename: str) -> dict:
    img = Image.open(filename)
    width, height = img.size
    return {
        'width': width,
        'height': height
    }


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
