import pandas as pd

def select_features_ohe(X: pd.DataFrame) -> list:
    feat_categorical_nunique = X.select_dtypes(include='object').nunique()
    feat_categorical_small = feat_categorical_nunique[feat_categorical_nunique<7]
    return feat_categorical_small.index.tolist()
