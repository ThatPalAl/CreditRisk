from pathlib import Path
import pandas as pd

IN_PATH  = Path("data/german_credit_data/credit_data_ready.csv")
OUT_PATH = Path("data/german_credit_data/credit_data_validated.csv")

df = pd.read_csv(IN_PATH)
print("dataset shape:", df.shape)
print("dataset columns:", list(df.columns))

assert "target" in df.columns, "No 'target' column in credit_data_ready.csv"
print("target values (as loaded):", df["target"].value_counts().to_dict())

if "age_group" in df.columns:
    df.drop(columns=["age_group"], inplace=True)
    print("Dropping helper column (only used for eda): 'age_group'")

non_num_cols = df.drop(columns=["target"]).select_dtypes(exclude=["number"]).columns.tolist()
if non_num_cols:
    print("Found non-numeric feature columns:", non_num_cols)

    for col in list(non_num_cols):
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() > 0.99:
            df[col] = coerced.fillna(0).astype("int8")
    non_num_cols = df.drop(columns=["target"]).select_dtypes(exclude=["number"]).columns.tolist()

    if non_num_cols:
        for col in non_num_cols:
            codes, uniques = pd.factorize(df[col].astype("string"), sort=True)
            df[col] = codes.astype("int32")
        print("Factorized remaining categoricals:", non_num_cols)
else:
    print("All feature columns already numeric.")

num = df.drop(columns=["target"]).select_dtypes(include=["number"])
assert num.shape[1] == df.shape[1]-1, "Still non-numeric features remain after conversion."
assert df.isna().sum().sum() == 0, "NaNs present after conversion!"

print("numeric features:", num.shape[1])
print("Dataset validated & numeric.")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")