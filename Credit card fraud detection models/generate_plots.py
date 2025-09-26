import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore

file_path = "credit_card_train.csv"   
df = pd.read_csv(file_path)

num_cols = df.select_dtypes(include=np.number).columns

ncols = 4 
nrows = int(np.ceil(len(num_cols) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
    axes[i].set_title(col)

for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(12,10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,cbar=True, annot_kws={"size": 6})

plt.title("Correlation Matrix")
plt.savefig("correlations.png", dpi=150)
plt.close()
