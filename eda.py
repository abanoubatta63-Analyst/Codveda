"""
Codveda Data Analytics Internship
Level 1 - Task 2: Exploratory Data Analysis (EDA)
Dataset: Iris Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -----------------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------------
df = pd.read_csv("1__iris.csv")
print("Shape:", df.shape)
print(df.head())

numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# -----------------------------------------------------------
# 2. Summary statistics
# -----------------------------------------------------------
summary = df[numeric_cols].describe().T
summary["mode"] = df[numeric_cols].mode().iloc[0]
print("\nSummary statistics:\n", summary)

summary.to_csv("summary_statistics.csv")

# -----------------------------------------------------------
# 3. Distributions - Histograms
# -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("histograms.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 4. Distributions - Boxplots (by species)
# -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(data=df, x="species", y=col, ax=ax, hue="species", legend=False)
    ax.set_title(f"{col} by species")
plt.tight_layout()
plt.savefig("boxplots.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 5. Scatter plots - relationships between features
# -----------------------------------------------------------
pairplot = sns.pairplot(df, hue="species", corner=True)
pairplot.savefig("scatter_pairplot.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 6. Correlation between numerical features
# -----------------------------------------------------------
corr = df[numeric_cols].corr()
print("\nCorrelation matrix:\n", corr)

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - Iris Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

print("\nEDA complete. Files saved: summary_statistics.csv, histograms.png, "
      "boxplots.png, scatter_pairplot.png, correlation_heatmap.png")
