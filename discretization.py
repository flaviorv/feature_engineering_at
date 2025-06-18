from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

breast_cancer = load_breast_cancer()
features = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
selected = ['mean radius', 'mean texture']
kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
fixed = kbd.fit_transform(features[selected])
fixed = pd.DataFrame(fixed, columns=['fixed '+selected[0], 'fixed '+selected[1]])
kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
varied = kbd.fit_transform(features[selected])
varied = pd.DataFrame(varied, columns=['varied '+selected[0], 'varied '+selected[1]])

all = pd.concat([fixed, varied], axis=1)
print(f'Continuous features:\n{features}')
print(f'\nDicretization\n{all.apply(pd.Series.value_counts)}')
