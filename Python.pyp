import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ans
titanic = sns.load_dataset("titanic")

print("Shape of data:", titanic.shape)
print("\nFirst 5 rows:\n", titanic.head())

print("\nMissing values:\n", titanic.isnull().sum())

titanic["age"].fillna(titanic["age"].mean(), inplace=True)

titanic["embark_town"].fillna(titanic["embark_town"].mode()[0], inplace=True)

titanic.drop(columns=["deck"], inplace=True)

print("\nSummary stats:\n", titanic.describe())

plt.figure(figsize=(6,4))
sns.countplot(x="sex", data=titanic)
plt.title("Count of Male vs Female")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(titanic["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="class", y="age", data=titanic)
plt.title("Age by Passenger Class")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x="sex", y="survived", data=titanic)
plt.title("Survival Rate by Gender")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("\nInsights:")
print("- Females had higher survival rate than males.")
print("- Younger passengers survived more often than older.")
print("- Higher class (1st class) had better survival chances.")
