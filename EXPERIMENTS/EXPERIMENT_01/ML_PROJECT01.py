import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# --- 1. Import the dataset ---
# NOTE: This assumes 'Employee.csv' is in the same directory as the script.
try:
    df = pd.read_csv('Employee.csv')
    print(f"Initial shape of the dataset: {df.shape}")
except FileNotFoundError:
    print("Error: Employee.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Create a copy for cleaning operations
df_cleaned = df.copy()

# --- 2. Data Exploration and Cleaning (Duplicates) ---
# Remove duplicate rows
initial_rows = df_cleaned.shape[0]
df_cleaned.drop_duplicates(inplace=True)
df_cleaned.reset_index(drop=True, inplace=True)
print(f"Removed {initial_rows - df_cleaned.shape[0]} duplicate rows.")
print(f"Shape after removing duplicates: {df_cleaned.shape}")

# --- 3. Cleaning Inconsistent Categorical Values (Corrected inplace issues) ---

# Dictionary for Company replacements
company_replacements = {
    'Infosys Pvt Lmt': 'Infosys',
    'Tata Consultancy Services': 'TCS',
    # Note: Chaining the CTS -> Congnizant -> Cognizant should be done in one go
    # to avoid intermediate misspellings. The dict handles this correctly.
    'CTS': 'Cognizant',
    'Congnizant': 'Cognizant'
}
# Corrected replacement using direct assignment (avoids FutureWarning)
df_cleaned['Company'] = df_cleaned['Company'].replace(company_replacements)
print("\nCompany unique values after cleaning:")
print(df_cleaned['Company'].unique())

# Corrected replacement for 'Place' (avoids FutureWarning)
df_cleaned['Place'] = df_cleaned['Place'].replace('Podicherry', 'Pondicherry')
print("\nPlace unique values after cleaning:")
print(df_cleaned['Place'].unique())

# --- 4. Handling Missing Values and Outliers ---
print("\nMissing values before imputation:")
print(df_cleaned.isna().sum())

# A. Treat 'Age' Outlier (Age=0) by replacing with NaN
# Corrected replacement (avoids FutureWarning)
df_cleaned['Age'] = df_cleaned['Age'].replace(0, np.nan)
print("\nMissing values after changing Age=0 to NaN:")
print(df_cleaned.isna().sum())

# B. Impute 'Company' (Categorical NaN): Fill with Mode
company_mode = df_cleaned['Company'].mode()[0]
# Corrected fillna (avoids FutureWarning)
df_cleaned['Company'] = df_cleaned['Company'].fillna(company_mode)
print(f"\nImputed Company NaN with mode: {company_mode}")

# C. Impute 'Age' (Numerical NaN): Fill with Mean
# Calculate mean after Age=0 replacement
rounded_age_mean = round(df_cleaned['Age'].mean(), 0)
df_cleaned['Age'] = df_cleaned['Age'].fillna(rounded_age_mean)
print(f"Imputed Age NaN with rounded mean: {rounded_age_mean}")

# D. Impute 'Salary' (Numerical NaN): Fill with Mean
rounded_salary_mean = round(df_cleaned['Salary'].mean(), 0)
df_cleaned['Salary'] = df_cleaned['Salary'].fillna(rounded_salary_mean)
print(f"Imputed Salary NaN with rounded mean: {rounded_salary_mean}")

# E. Impute 'Place' (Categorical NaN): Fill with Mode
place_mode = df_cleaned['Place'].mode()[0]
# Corrected fillna (avoids FutureWarning)
df_cleaned['Place'] = df_cleaned['Place'].fillna(place_mode)
print(f"Imputed Place NaN with mode: {place_mode}")

print("\nFinal missing value count:")
print(df_cleaned.isna().sum())

# F. Convert Gender 0/1 to M/F for better encoding preparation
df_cleaned['Gender'] = df_cleaned['Gender'].replace({0: 'M', 1: 'F'})

# Drop 'Country' as it has only one unique value ('India')
df_cleaned.drop('Country', axis=1, inplace=True)

print("\nCleaned and pre-processed DataFrame head:")
print(df_cleaned.head())
print("\nCleaned DataFrame info:")
df_cleaned.info()


# --- 5. Data Encoding (OneHotEncoding) ---

# Define categorical features to encode
categorical_features = ['Company', 'Place', 'Gender']

# Initialize OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

# Fit and transform the categorical data
df_array = ohe.fit_transform(df_cleaned[categorical_features])

# Get feature names for the new columns
feature_names = ohe.get_feature_names_out(categorical_features)

# Create a DataFrame for the encoded data
df_new = pd.DataFrame(df_array, columns=feature_names)

# Concatenate the original DataFrame (without the categorical columns) and the new encoded DataFrame
df_ml = pd.concat([df_cleaned.drop(categorical_features, axis=1), df_new], axis=1)

print("\n--- DataFrame after One-Hot Encoding ---")
print(df_ml.head())


# --- 6. Feature Scaling (StandardScaler) ---

# Select the numerical columns to scale
data_to_scale = df_ml[['Age', 'Salary']]

# Initialize and fit StandardScaler
scaler_ss = StandardScaler()
data_scaled_ss = scaler_ss.fit_transform(data_to_scale)

# Convert scaled array back to DataFrame for verification
scaled_data_set_ss = pd.DataFrame(data_scaled_ss, columns=data_to_scale.columns)
print("\n--- Age and Salary Scaled using StandardScaler (Z-Score) ---")
print(scaled_data_set_ss.describe())


# --- 7. Feature Scaling (MinMaxScaler) ---

# Initialize and fit MinMaxScaler
scaler_mm = MinMaxScaler()
data_scaled_mm = scaler_mm.fit_transform(data_to_scale)

# Convert scaled array back to DataFrame for verification
mm_scaled_data_set = pd.DataFrame(data_scaled_mm, columns=data_to_scale.columns)
print("\n--- Age and Salary Scaled using MinMaxScaler (0-1 Range) ---")
print(mm_scaled_data_set.describe())

# Save the fully cleaned dataset (pre-encoded)
df_cleaned.to_csv('Employee_cleaned.csv', index=False)
print("\nSaved the cleaned dataset to 'Employee_cleaned.csv'")