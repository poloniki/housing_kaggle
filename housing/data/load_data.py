import pandas as pd

def load_data():
    data = pd.read_csv("https://wagon-public-datasets.s3.amazonaws.com/houses_train_raw.csv")
    X = data.drop(columns=['SalePrice', 'Id'])
    y = data['SalePrice']
    return X, y
