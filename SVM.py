# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm, metrics

# Loading Dataset
col_names = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year', 'Brand']
data = pd.read_csv("cars.csv", header=None, names=col_names)


# Splitting Dataset in features and target variable
feature_cols = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year']
X = data[feature_cols]
Y = data.Brand  # Target Variable | Target Names = ['US','Japan']

plt.figure(figsize=(10, 10))
plt.scatter(X[0:1], X[1:2], marker='*', s=100, edgecolors='black')
plt.show()

# Creating test set and training set from general dataset %70 for training %30 for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Calculate accuracy

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))



