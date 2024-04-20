# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
col_names = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year', 'Brand']
data = pd.read_csv("cars.csv", header=None, names=col_names)


# Splitting Dataset into features and target variable
feature_cols = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year']
X = data[feature_cols]
Y = data.Brand  # Target Variable | Target Names = ['US','Japan']

# Creating a scatter plot (just for visualization purposes)
plt.figure(figsize=(10, 10))
plt.scatter(X['MPG'], X['Weight'], marker='*', s=100, edgecolors='black')
plt.show()

# Creating test set and training set from the general dataset (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Train the model using the training sets
knn.fit(X_train_scaled, y_train)

# Predict the response for the test dataset
y_pred = knn.predict(X_test_scaled)

# Calculate accuracy

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


