# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
# MODIFIED: Load the local 'Social_Network_Ads.csv' file
data = pd.read_csv("Social_Network_Ads.csv")

# Display first 5 rows
print("Sample Data:\n", data.head())

# Step 3: Check for Missing Values
print("\nMissing Values:\n", data.isnull().sum())

# --- NEW PREPROCESSING STEPS ---
# Drop 'User ID' as it's not a predictive feature
data = data.drop('User ID', axis=1)

# Convert 'Gender' to a numeric format (e.g., 0 for Female, 1 for Male)
# Using drop_first=True to avoid multi-collinearity (dummy variable trap)
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

print("\nData after preprocessing (Gender encoded):\n", data.head())
# -------------------------------

# Step 4: Separate Features and Target
# MODIFIED: Use 'Purchased' as the target and the rest as features
X = data.drop('Purchased', axis=1) # input features
y = data['Purchased'] # output labels

# Print feature names to be sure
print("\nFeatures (X) columns:\n", X.columns.tolist())

# Step 5: Split Data into Training and Testing Sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling (Standardization)
# This is very important here as Age and EstimatedSalary are on different scales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Define and Train the SVM Model
# The model definition remains the same
# We are solving a binary classification problem (Purchased 0 or 1)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

# Step 8: Make Predictions on Test Data
y_pred = svm_model.predict(X_test_scaled)

# Step 9: Evaluate the Model
# The evaluation metrics are the same for this binary classification task
print("\nModel Evaluation Results:")
# Calculate and print accuracy
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Print the classification report (precision, recall, f1-score)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Test on a New Sample
# MODIFIED: New sample must match the features: [Age, EstimatedSalary, Gender_Male]
# Let's predict for a 35-year-old Male (1) with an $85,000 salary
sample = np.array([[35, 85000, 1]]) 
# Note: The order must match X.columns: ['Age', 'EstimatedSalary', 'Gender_Male']

# Scale the new sample using the same scaler
sample_scaled = scaler.transform(sample)

# Predict the purchase
prediction = svm_model.predict(sample_scaled)
prediction_label = 'Purchased' if prediction[0] == 1 else 'Not Purchased'
print(f"\nPrediction for new sample (35yo Male, $85k Salary): {prediction_label} (Class: {prediction[0]})")