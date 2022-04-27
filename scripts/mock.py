import pandas as pd


df = pd.read_csv("data/raw/dataset.csv", sep=";")

print("---> Head:", df.head(), sep="\n")
print("---> Missing default info:", df.default.isnull().mean(), sep="\n")
print("---> Available default info:", df.default.value_counts(normalize=True), sep="\n")
