# --- Step 1: Import Required Libraries ---
# (from Experiment 7, Page 2)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing # Added to load a runnable dataset

# --- Step 2: Load Dataset ---
# Using California Housing dataset as a replacement for 'your_dataset.csv'
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target # Using 'PRICE' as the target column

print("--- Dataset Loaded (California Housing) ---")
print(data.head())
print("\n")

# --- Step 3: Separate Features and Target ---
# (from Experiment 7, Page 2)
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# --- Step 4: Split Data into Training and Testing Sets ---
# (from Experiment 7, Page 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 5: Feature Scaling ---
# (from Experiment 7, Page 3)
# Scaling X features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scaling y target (needed for SVR)
scaler_y = StandardScaler()
# Reshape y_train to 2D array for scaling, then flatten back with .ravel()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# --- Step 6: Define Models ---
# (from Experiment 7, Page 3)
models = {
    "Linear Regression": LinearRegression(),
    # Polynomial Regression uses a LinearRegression model on transformed features
    "Polynomial Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf')
}

# --- Step 7: Polynomial Feature Transformation ---
# (from Experiment 7, Page 3)
# Create 2nd-degree polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# --- Step 8: Train and Evaluate Models ---
# (from Experiment 7, Page 3-4)
results = {}

print("--- Training and Evaluating Models ---")

for name, model in models.items():
    if name == "Polynomial Regression":
        # Train on polynomial features
        model.fit(X_poly_train, y_train)
        y_pred = model.predict(X_poly_test)
        
    elif name == "SVR":
        # Train on scaled X and scaled y
        model.fit(X_train_scaled, y_train_scaled)
        # Predict on scaled X, then inverse_transform the scaled predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
    else:
        # Train on original (unscaled) X for tree-based models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Store evaluation metrics
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    }
    print(f"Completed: {name}")


# --- Step 9: Display Results ---
# (from Experiment 7, Page 4)
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="R2_Score", ascending=False) # Sort by best R2 score

print("\n--- Model Comparison Results ---")
print(results_df)
print("\n--- Conclusion ---")
print(f"The best performing model is: {results_df.index[0]} (R2 Score: {results_df.iloc[0, 2]:.4f})")