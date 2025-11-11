# ================================================================
# Assignment 1 — Statistical Methods in AI
# ================================================================
# Q1: Student Dataset Generation and Visualization
# Q2: KNN Classification
# Q3: Polynomial Regression with Regularization
# ================================================================

# --- Imports and Setup ---
import hashlib
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# USERNAME and SEED
# ================================================================
username = "samyak.kamble@gcekarad.ac.in"   # replace with your IIITH email
_seed = int(hashlib.sha256(username.encode()).hexdigest(), 16) % (2 ** 32)

plt.rcParams['figure.figsize'] = (8, 5)
sns.set(style='whitegrid')


# ================================================================
# Q1: Student Dataset Generation, Visualization and Sampling
# ================================================================
class StudentDataset:
    def __init__(self, num_students: int = 10000, seed: int = None):
        self.num_students = int(num_students)
        self.seed = seed if seed is not None else 0
        self.rng = np.random.default_rng(self.seed)
        self._df = self.assemble_dataframe()

    def get_full_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def generate_gender(self) -> List[str]:
        probs = [0.65, 0.33, 0.02]
        choices = ['Male', 'Female', 'Other']
        return self.rng.choice(choices, size=self.num_students, p=probs).tolist()

    def generate_major(self) -> List[str]:
        probs = [0.70, 0.20, 0.10]
        choices = ['B.Tech', 'MS', 'PhD']
        return self.rng.choice(choices, size=self.num_students, p=probs).tolist()

    def generate_program(self, majors: List[str]) -> List[str]:
        programs = []
        for maj in majors:
            if maj == 'B.Tech':
                probs = [0.4, 0.4, 0.1, 0.1]
            elif maj == 'MS':
                probs = [0.3, 0.3, 0.2, 0.2]
            else:
                probs = [0.25, 0.25, 0.25, 0.25]
            choices = ['CSE', 'ECE', 'CHD', 'CND']
            programs.append(self.rng.choice(choices, p=probs))
        return programs

    def generate_gpa(self, majors: List[str]) -> List[float]:
        gpas = []
        for maj in majors:
            if maj == 'B.Tech':
                mu, sigma = 7.0, 1.0
            elif maj == 'MS':
                mu, sigma = 8.0, 0.7
            else:
                mu, sigma = 8.3, 0.5
            val = self.rng.normal(loc=mu, scale=sigma)
            gpas.append(float(np.clip(val, 4.0, 10.0)))
        return gpas

    def assemble_dataframe(self) -> pd.DataFrame:
        genders = self.generate_gender()
        majors = self.generate_major()
        programs = self.generate_program(majors)
        gpas = self.generate_gpa(majors)
        df = pd.DataFrame({
            'gender': genders,
            'major': majors,
            'program': programs,
            'GPA': gpas
        })
        df.index.name = 'student_id'
        df.reset_index(inplace=True)
        df['username'] = username
        return df

    # --- Visualization ---
    def plot_gender_distribution(self):
        df = self.get_full_dataframe()
        sns.countplot(x='gender', data=df, order=['Male', 'Female', 'Other'])
        plt.title('Gender distribution - ' + username)
        plt.show()

    def plot_major_distribution(self):
        df = self.get_full_dataframe()
        sns.countplot(x='major', data=df, order=['B.Tech', 'MS', 'PhD'])
        plt.title('Major distribution - ' + username)
        plt.show()

    def plot_program_by_major(self):
        df = self.get_full_dataframe()
        ct = pd.crosstab(df['major'], df['program'])
        ct.plot(kind='bar', stacked=True)
        plt.title('Program conditioned on Major - ' + username)
        plt.show()

    def gpa_mean_std(self) -> Tuple[float, float]:
        df = self.get_full_dataframe()
        return float(df['GPA'].mean()), float(df['GPA'].std())


# --- Run Q1 ---
if __name__ == "__main__":
    ds = StudentDataset(num_students=10000, seed=_seed)
    df = ds.get_full_dataframe()

    print("================================================")
    print("Q1: Student Dataset Head")
    print("================================================")
    print(df.head())

    print("\nGPA Mean and Std:")
    print(ds.gpa_mean_std())

    ds.plot_gender_distribution()
    ds.plot_major_distribution()
    ds.plot_program_by_major()

    # ================================================================
    # Q2: K-Nearest Neighbors (KNN) Classification
    # ================================================================
    X = df[['gender', 'program', 'GPA']]
    y = df['major']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_seed
    )

    categorical_features = ['gender', 'program']
    numeric_features = ['GPA']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

    knn = Pipeline([
        ('pre', preprocessor),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n================================================")
    print("Q2: KNN Classification Results")
    print("================================================")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ================================================================
    # Q3: Polynomial Regression with Regularization (L1/L2)
    # ================================================================
    np.random.seed(_seed)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    # Corrected polynomial: cubic relationship with noise
    y = 0.5 * (X ** 3) - 4 * (X ** 2) + 3 * X + 10 + np.random.normal(0, 10, X.shape)

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    ridge = Ridge(alpha=10)
    lasso = Lasso(alpha=0.5)
    linreg = LinearRegression()

    ridge.fit(X_poly, y)
    lasso.fit(X_poly, y)
    linreg.fit(X_poly, y)

    y_pred_ridge = ridge.predict(X_poly)
    y_pred_lasso = lasso.predict(X_poly)
    y_pred_lin = linreg.predict(X_poly)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='gray', label='Data')
    plt.plot(X, y_pred_lin, label='Linear Regression', linewidth=2)
    plt.plot(X, y_pred_ridge, label='Ridge (L2)', linewidth=2)
    plt.plot(X, y_pred_lasso, label='Lasso (L1)', linewidth=2)
    plt.legend()
    plt.title('Q3: Polynomial Regression with Regularization - ' + username)
    plt.show()

    print("\nAssignment 1 completed successfully ✅")
