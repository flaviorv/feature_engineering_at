from seaborn import load_dataset
import pandas as pd

penguins = load_dataset('penguins').dropna()
categorical_features = ['species', 'island', 'sex']
vector_space = pd.get_dummies(penguins, columns=categorical_features, prefix='specie', dtype=int)
scalar = penguins['bill_length_mm'][0]

print('Scalar:', scalar)
print('Vector:', list(vector_space.iloc[1, :]))
print(f'Vector Space:\n{vector_space.to_numpy()}')