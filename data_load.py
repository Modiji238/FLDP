import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_partition(client_id, num_clients, path="data/raw/creditcard.csv"):
    df = pd.read_csv(path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    size = len(X) // num_clients
    start = client_id * size
    end = start + size

    X_part = X[start:end]
    y_part = y[start:end]

    return train_test_split(X_part, y_part, test_size=0.2, random_state=42)