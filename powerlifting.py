import pandas as pd
from sklearn.feature_extraction import FeatureHasher, DictVectorizer

data = pd.read_csv('./archive/openpowerlifting.csv').dropna()
categorics = ['Equipment', 'Sex', 'Division']
print(data.iloc[0, :])

one_hot = pd.get_dummies(data[categorics], drop_first=False, dtype=int)
print("\nOne-Hot Encoding:")
print(one_hot.head())
print(one_hot.shape)

dummy = pd.get_dummies(data[categorics], drop_first=True, dtype=int)
print("\nDummy Coding:")
print(dummy.head())
print(dummy.shape)

def effect_coding(series):
    dummies = pd.get_dummies(series, dtype=int)
    dummies = dummies.iloc[:, :-1]
    dummies = dummies.apply(lambda col: col.replace(0, -1))
    return dummies

sex_eff = effect_coding(data['Sex']).add_prefix('Sex_')
event_eff = effect_coding(data['Division']).add_prefix('Division_')
equip_eff = effect_coding(data['Equipment']).add_prefix('Equip_')
effect = pd.concat([sex_eff, event_eff, equip_eff], axis=1)
print("\nEffect Coding:")
print(effect.head())
print(effect.shape)

cat_dicts = data[categorics].astype(str).to_dict(orient='records')

hasher = FeatureHasher(n_features=10, input_type='dict')
hashed = hasher.fit_transform(cat_dicts)
print("\nFeature hashing")
print(hashed.toarray())
print(hashed.shape)

vectorizer = DictVectorizer(sparse=False) 
bin_counted = vectorizer.fit_transform(cat_dicts)
print("\nBin Counting")
print(bin_counted)
print(bin_counted.shape)

