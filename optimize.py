from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
import pandas as pd

df = pd.read_csv("features.csv")

# Separate features and target from your dataframe
X = df.drop('label', axis=1)  # assuming 'label' is your target column
y = df['label']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include='number')),
        ('cat', OneHotEncoder(), make_column_selector(dtype_include='object'))
    ])

# Define your pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('reducer', TruncatedSVD(n_components=10))  # reduce to 10 components
])

# Fit and transform the data
X_transformed = pipeline.fit_transform(X)

# Create a new dataframe with the transformed features
df_transformed = pd.DataFrame(X_transformed, columns=[f'feature_{i}' for i in range(10)])

# Add back the target column
df_transformed['label'] = y

df_transformed.to_csv('features_reduced.csv', index=False)
