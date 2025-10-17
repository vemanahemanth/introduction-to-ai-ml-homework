import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- Task 1: Exploratory Data Analysis (EDA) ---
print("--- Task 1: EDA ---")

# Load the Titanic dataset
try:
    df = sns.load_dataset('titanic')
    print("Titanic dataset loaded from Seaborn.")
except Exception as e:
    print(f"Error loading dataset. {e}. (Need internet connection?)")
    # You would load your local CSV here if needed

# Summarize missing values and data types
print("\nData Info (Missing Values & Types):")
print(df.info())

# Visualize distributions of key features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['age'].dropna(), kde=True, ax=axes[0, 0]).set_title('Age Distribution')
sns.countplot(data=df, x='sex', ax=axes[0, 1]).set_title('Sex Distribution')
sns.countplot(data=df, x='pclass', ax=axes[1, 0]).set_title('Pclass Distribution')
sns.histplot(df['fare'], kde=True, ax=axes[1, 1]).set_title('Fare Distribution')
plt.tight_layout()
plt.show()

# Analyze relationships with survival
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=df, x='sex', y='survived', ax=axes[0]).set_title('Survival by Sex')
sns.barplot(data=df, x='pclass', y='survived', ax=axes[1]).set_title('Survival by Pclass')
plt.show()


# --- Task 2: Data Cleaning and Imputation ---
print("\n--- Task 2: Data Cleaning & Imputation ---")

# Handle missing Age (impute with median)
age_median = df['age'].median()
df['age'] = df['age'].fillna(age_median)
print(f"Missing 'age' imputed with median: {age_median}")

# Handle missing Embarked (impute with mode)
embarked_mode = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)
print(f"Missing 'embarked' imputed with mode: {embarked_mode}")

# Handle missing Fare (impute with median)
fare_median = df['fare'].median()
df['fare'] = df['fare'].fillna(fare_median)

# Drop irrelevant columns (PassengerId, Name, Ticket, Cabin)
# Note: Seaborn dataset doesn't have PassengerId
# We keep 'name' temporarily for feature engineering
# 'deck' is the same as 'cabin' in this dataset and has too many nulls
df = df.drop(['deck', 'embark_town'], axis=1)
print("Dropped 'deck' and 'embark_town' columns.")


# --- Task 3: Feature Engineering ---
print("\n--- Task 3: Feature Engineering ---")

# Create FamilySize
df['FamilySize'] = df['sibsp'] + df['parch'] + 1
print("Created 'FamilySize' feature.")

# Extract Titles from Name
df['Title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Consolidate titles
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
print(f"Extracted and consolidated 'Title' feature. Titles: {df['Title'].unique()}")

# Convert categorical features (Sex, Embarked, Title)
df = pd.get_dummies(df, columns=['sex', 'embarked', 'Title'], drop_first=True)
print("Converted categorical features using one-hot encoding.")

# Drop original columns no longer needed
df = df.drop(['name', 'ticket', 'sibsp', 'parch', 'class', 'who', 'adult_male', 'alive', 'alone'], 
             axis=1, errors='ignore')
print("Dropped original 'name', 'ticket', 'sibsp', 'parch' and other redundant columns.")


# --- Task 4: Prepare Data for Modeling ---
print("\n--- Task 4: Prepare Data for Modeling ---")

# Finalize features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check data readiness
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("\nData is ready for modeling. X_train head:")
print(X_train.head())