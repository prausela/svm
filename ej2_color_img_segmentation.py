import pandas as pd
from sklearn.svm import SVC
from utils.img_utils import img_to_classified_df, img_to_df
from utils.data_split import k_fold_split
from utils.sklearn_utils import c_to_sklearn_c

def run_ej2():
    cow_df = img_to_classified_df("data/vaca.jpg", 'cow')
    grass_df = img_to_classified_df("data/pasto.jpg", 'grass')
    sky_df = img_to_classified_df("data/cielo.jpg", 'sky')
    df = pd.concat([cow_df, grass_df, sky_df], ignore_index=True)
    sample_df = img_to_df("data/cow.jpg")
    train_df, test_df = k_fold_split(df, k=5)

    X = train_df[["R", "G", "B"]].to_numpy()
    Y = train_df["Class"].to_numpy()

    sk_c = c_to_sklearn_c(1)

    svm = SVC(C=sk_c, kernel="rbf")
    svm.fit(X, Y)

if __name__ == '__main__':
    run_ej2()
