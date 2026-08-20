"""
Codveda Data Analytics Internship
Level 1 - Task 1: Data Cleaning and Preprocessing
Dataset: Social Media Sentiment Dataset
"""

import pandas as pd

# -----------------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------------
df = pd.read_csv("3__Sentiment_dataset.csv")

print("Original shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)

# -----------------------------------------------------------
# 2. Drop redundant / unnecessary columns
#    'Unnamed: 0' and 'Unnamed: 0.1' are leftover index columns
# -----------------------------------------------------------
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")

# -----------------------------------------------------------
# 3. Identify and handle missing values
# -----------------------------------------------------------
print("\nMissing values per column:\n", df.isnull().sum())

# Numeric columns -> fill missing with median (robust to outliers)
numeric_cols = df.select_dtypes(include="number").columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Text/categorical columns -> fill missing with 'Unknown'
text_cols = df.select_dtypes(include="object").columns
for col in text_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna("Unknown")

# -----------------------------------------------------------
# 4. Standardize inconsistent text formats
#    Many text columns have leading/trailing whitespace
#    e.g. " Positive  " vs "Positive" are currently treated
#    as different categories
# -----------------------------------------------------------
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# Standardize casing for key categorical fields
df["Sentiment"] = df["Sentiment"].str.title()
df["Platform"] = df["Platform"].str.title()
df["Country"] = df["Country"].str.title()

# -----------------------------------------------------------
# 5. Fix inconsistent date format
# -----------------------------------------------------------
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# -----------------------------------------------------------
# 6. Remove duplicate rows
# -----------------------------------------------------------
dupes_before = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\nDuplicate rows removed: {dupes_before}")

# -----------------------------------------------------------
# 7. Clean numeric columns (ensure correct types, no negatives)
# -----------------------------------------------------------
for col in ["Retweets", "Likes"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].clip(lower=0)  # engagement counts can't be negative

# -----------------------------------------------------------
# 8. Final check
# -----------------------------------------------------------
print("\nFinal shape:", df.shape)
print("\nUnique Sentiment values after cleaning:\n", df["Sentiment"].unique())
print("\nMissing values after cleaning:\n", df.isnull().sum().sum())

# -----------------------------------------------------------
# 9. Save cleaned dataset
# -----------------------------------------------------------
df.to_csv("sentiment_dataset_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'sentiment_dataset_cleaned.csv'")
