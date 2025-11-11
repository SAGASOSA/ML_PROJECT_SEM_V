# --- Step 1: Import Required Libraries ---
# (from exp 10.pdf, Cell 1)
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 2: Load Dataset ---
# (from exp 10.pdf, Cell 2)
iris = load_iris()
X = iris.data
y = iris.target

# --- Step 3: Simplify to a Binary Classification Problem ---
# (from exp 10.pdf, Cell 3)
# Remove class 2 (Virginica) to make it a binary problem (Setosa vs. Versicolor)
X_simple = X[y != 2]
y_simple = y[y != 2]

# --- Step 4: Split Data into Training and Testing Sets ---
# (from exp 10.pdf, Cell 4)
X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=42, stratify=y_simple
)

# --- Step 5 & 6: Feature Scaling ---
# (from exp 10.pdf, Cell 5 & 6)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 7: Define and Train the Perceptron Model ---
# (from exp 10.pdf, Cell 7)
perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
perceptron.fit(X_train_scaled, y_train)

# --- Step 8: Make Predictions and Evaluate the Model ---
# (from exp 10.pdf, Cell 8)
y_pred = perceptron.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\n--- Classification Report ---")
# Use the target names for the two classes we kept (0 and 1)
print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor']))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- Step 9: Visualize the Confusion Matrix ---
# (from exp 10.pdf, Cell 8)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Setosa', 'Versicolor'],
            yticklabels=['Setosa', 'Versicolor'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()