import os
import subprocess

DATASET = "mlg-ulb/creditcardfraud"

def download():
    os.makedirs("data/raw", exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET,
        "-p", "data/raw",
        "--unzip"
    ])

if __name__ == "__main__":
    download()
    print("Dataset downloaded successfully")
