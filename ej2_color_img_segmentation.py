import pandas as pd
from sklearn import svm
from utils.img_utils import img_to_classified_df, img_to_df
from utils.data_split import k_fold_split

def run_ej2():
    cow_df = img_to_classified_df("data/vaca.jpg", 'cow')
    grass_df = img_to_classified_df("data/pasto.jpg", 'grass')
    sky_df = img_to_classified_df("data/cielo.jpg", 'sky')
    df = pd.concat([cow_df, grass_df, sky_df], ignore_index=True)
    # print(df)
    sample_df = img_to_df("data/cow.jpg")
    train_df, test_df = k_fold_split(df, k=5)

if __name__ == '__main__':
    run_ej2()
