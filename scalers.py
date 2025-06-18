from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, FunctionTransformer
import pandas as pd
import math

breast_cancer = load_breast_cancer()
features = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
print(f'Raw\n{features}')

min_max = MinMaxScaler()
scaled_minmax = min_max.fit_transform(features)
scaled_minmax = pd.DataFrame(scaled_minmax, columns=features.columns)
print(f'\nMin Max\n{scaled_minmax}')
print(f'mean radius min: {scaled_minmax['mean radius'].min()} max: {scaled_minmax['mean radius'].max()} ')

std = StandardScaler()
scaled_std = std.fit_transform(features)
scaled_std = pd.DataFrame(scaled_std, columns=features.columns)
print(f'\nStandard Scaler\n{scaled_std}')
print(f'mean radius average {round(scaled_std['mean radius'].mean(), 2)}')
print(f'mean radius deviation {round(scaled_std['mean radius'].std(), 2)}')

l2 = Normalizer(norm='l2')
scaled_l2 = l2.fit_transform(features)
row = scaled_l2[1]
sum = 0
for col in row:
    col = col**2
    sum+=col
scaled_l2 = pd.DataFrame(scaled_l2, columns=features.columns)
print(f'\nL2\n{scaled_l2}')
print(f'Magnitude of a sample {math.sqrt(sum)}')