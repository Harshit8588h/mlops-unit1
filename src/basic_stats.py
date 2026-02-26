import pandas as pd

def load_and_describe(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())
    print("\nBasic Statistics:\n", df.describe())

if __name__ == "__main__":
    load_and_describe("data/sample.csv")