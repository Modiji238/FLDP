import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_DIR = "copies"

files = [f for f in os.listdir(RESULT_DIR) if f.endswith(".csv")]

experiments = {}

for file in files:
    path = os.path.join(RESULT_DIR, file)
    df = pd.read_csv(path)

    name = file.replace(".csv", "")
    experiments[name] = df


metrics = ["recall", "f1", "auc", "precision"]

for metric in metrics:

    plt.figure(figsize=(8,5))

    for name, df in experiments.items():
        plt.plot(df["round"], df[metric], label=name)

    plt.xlabel("Rounds")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs Rounds")
    plt.legend()
    plt.grid()

    plt.show()