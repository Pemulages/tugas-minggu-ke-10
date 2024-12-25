import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load and prepare data
df = pd.read_csv('Iris.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:")
print(df.describe())

# Drop ID column
df.drop(columns='Id', axis=1, inplace=True)

# Visualize species distribution
Species = df['Species'].value_counts().reset_index()
plt.figure(figsize=(8,8))
plt.pie(Species['count'], labels=['Iris-setosa','Iris-versicolor','Iris-virginica'], 
        autopct='%1.3f%%', explode=[0,0,0])
plt.legend(loc='upper left')
plt.title('Distribution of Iris Species')
plt.show()

# Sepal visualization
plt.figure(figsize=(10,5))
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=df, hue='Species')
plt.title('Sepal Width vs Sepal Length')
plt.show()

sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species",
           palette="bright", data=df)
plt.title("Sepal Length VS Sepal Width")
plt.show()

# Petal visualization
plt.figure(figsize=(10,5))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', data=df, hue='Species')
plt.title('Petal Width vs Petal Length')
plt.show()

sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", hue="Species",
           palette="bright", data=df)
plt.title("Petal Length VS Petal Width")
plt.show()

# Correlation analysis
newdf = df.drop(columns='Species', axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(newdf.corr(), annot=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Prepare data for modeling
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

X = df.drop(['Species'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

print("\nTraining set size:", len(X_train))
print("Test set size:", len(X_test))

# Train and evaluate KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print('\nModel Performance:')
print('Training accuracy: {:.2f}'.format(knn.score(X_train, y_train)))
print('Test accuracy: {:.2f}'.format(knn.score(X_test, y_test)))

# Optional: Make predictions
predictions = knn.predict(X_test)
print("\nPredictions on test set:")
print(predictions)