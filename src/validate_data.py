import pandas as pd

df = pd.read_csv("data/german_credit_data/credit_data_ready.csv")
print("dataset shape:", df.shape)
print("dataset columns :", list(df.columns))

assert "target" in df.columns, "No 'target' column in credit_data_ready.csv"
print("target values:", df["target"].value_counts().to_dict())

non_num_cols = df.drop(columns=["target"]).select_dtypes(exclude=["number"]).columns.tolist()

if non_num_cols:
    print("Found non-numeric feature columns:", non_num_cols)
    for col in non_num_cols:
        df[col], _ = pd.factorize(df[col])
    print(f"Converted non-numeric columns ({non_num_cols}) to numeric using factorize function")
else:
    print("all columns are numeric.")

num = df.drop(columns=["target"]).select_dtypes(include=["number"])

print("numeric features:", num.shape[1])
print("Dataset validated")
df.to_csv('data/german_credit_data/credit_data_validated.csv', index=False)