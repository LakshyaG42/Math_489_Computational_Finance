import pandas as pd
from sklearn.decomposition import PCA

file_name = "PCA_Treasuries.xlsx"
df = pd.read_excel(file_name, sheet_name="Daily data")
rates = df.iloc[:, 1:9]
rate_changes = rates.diff().dropna() * 100

pca_model = PCA(n_components=8)
pca_model.fit(rate_changes)
eig_vals = pca_model.explained_variance_

pc1_power = eig_vals[0] / eig_vals.sum()
pc1_3_power = eig_vals[:3].sum() / eig_vals.sum()

print("\nPrincipal Component Analysis (PCA) Results:")
for idx, eig in enumerate(eig_vals, start=1):
    print(f"Eigenvalue {idx}: {eig:.5f}")
print(f"\nVariance explained by PC1: {pc1_power:.2%}")
print(f"Variance explained by first 3 PCs: {pc1_3_power:.2%}")