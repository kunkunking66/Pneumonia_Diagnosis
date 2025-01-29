import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_process_data(file_path):
    data = pd.read_excel(file_path)
    data = data.dropna()
    X = data.drop(columns=['肺炎'])
    y = data['肺炎']
    X = pd.get_dummies(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)
