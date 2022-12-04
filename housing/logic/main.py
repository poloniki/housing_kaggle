from housing.data.load_data import load_data
from housing.data.preproccesing import select_features_ohe
from housing.logic.pipeline import create_pipeline

def main():
    X, y = load_data()
    ohe = select_features_ohe(X)
    pipeline = create_pipeline(ohe)
