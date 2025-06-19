from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./archive/Lung Cancer Dataset.csv')
print(data)

X = data.drop(columns=['PULMONARY_DISEASE'])
y = data['PULMONARY_DISEASE'] if 'PULMONARY_DISEASE' in data.columns else None
le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

var_exp = pca.explained_variance_ratio_
cum_var_exp = var_exp.cumsum()

plt.figure(figsize=(10,6))
plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.5, align='center', label='Individual Variance')
plt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where='mid', label='Cumulative Variance')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.savefig('explained_variance.png')

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y if y is not None else 'b', cmap='viridis', alpha=0.7)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Projection onto the First Two Principal Components')
plt.colorbar()
plt.savefig('principal_components.png')