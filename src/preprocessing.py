import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
df = pd.read_csv('data/vehicles.csv')  # ✅ Fixed path

# Step 2: Select useful columns
columns_to_keep = ['comb08', 'displ', 'cylinders', 'drive', 'trany', 'fuelType', 'VClass']
df = df[columns_to_keep]

# Step 3: Drop rows where target (comb08) is missing
df = df.dropna(subset=['comb08'])

# Step 4: Separate features by type
numeric_cols = ['displ', 'cylinders']
categorical_cols = ['drive', 'trany', 'fuelType', 'VClass']

# Step 5: Impute numeric features
num_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Step 6: One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 7: Save preprocessed dataset
df.to_csv('data/cleaned_vehicles.csv', index=False)

print("✅ Preprocessing complete. Cleaned data saved as 'data/cleaned_vehicles.csv'")