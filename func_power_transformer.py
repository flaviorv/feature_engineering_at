from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

breast_cancer = load_breast_cancer()
features = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
print('Without normalization')
print(features)

ft = FunctionTransformer(lambda x: np.cbrt(x))
normalized = ft.fit_transform(features)
print('\nNormalized with cubic root')
print(normalized)

pt = PowerTransformer()
transformed = pt.fit_transform(features)
transformed = pd.DataFrame(transformed, columns=breast_cancer.feature_names)
print('\nAfter PowerTransformer')
print(transformed)

features.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.savefig('distribution_before_pt')
transformed.hist(figsize=(12,8), bins=30)
plt.tight_layout()
plt.savefig('distribution_after_pt')
