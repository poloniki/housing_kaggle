from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_selector as selector, make_column_transformer


def create_pipeline(cat_features):
    """Create a pipeline that will be used to train the model."""
    num_features = selector(dtype_include=['int64', 'float64'])

    num_pipeline = make_pipeline(SimpleImputer(),
                                 MinMaxScaler())

    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                OneHotEncoder(handle_unknown='ignore'))

    preprocess = make_column_transformer((num_pipeline, num_features),
                                        (cat_pipeline, cat_features),
                                        remainder="drop")
    return preprocess
