# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv('titanic.csv') 
print("First 5 rows:")
print(df.head())

# Step 3: Explore Basic Information
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Handle Missing Values

# Fill missing Age with Median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with Mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fill missing Fare with Median (if any)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

print("\nMissing Values After Filling:")
print(df.isnull().sum())

# Step 5: Convert Categorical Features into Numeric (Encoding)

# Encoding 'Sex' column
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'])

print("\nDataset after Encoding:")
print(df.head())

# Step 6: Normalize Numerical Features (Optional)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['Fare'] = scaler.fit_transform(df[['Fare']])

print("\nFare column after Normalization:")
print(df['Fare'].head())

# Step 7: Visualize Outliers (Boxplots)

# Boxplot for Age
print("\nPlotting Boxplot for Age...")
sns.boxplot(x=df['Age'])
plt.title('Boxplot for Age')
plt.show()

# Boxplot for Fare
print("\nPlotting Boxplot for Fare...")
sns.boxplot(x=df['Fare'])
plt.title('Boxplot for Fare')
plt.show()

# Step 8: (Optional) Remove Outliers

# Remove extreme Age outliers (example)
df = df[df['Age'] <= 75]

print("\nShape of data after removing outliers:")
print(df.shape)