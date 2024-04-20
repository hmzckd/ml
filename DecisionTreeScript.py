# Importing necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

# Loading Dataset
col_names = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year', 'Brand']
data = pd.read_csv("../../Downloads/cars.csv", header=None, names=col_names)

# Printing first 5 row of the dataset
print(data.head())

# Splitting Dataset in features and target variable
feature_cols = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year']
X = data[feature_cols]
Y = data.Brand

# Creating test set and training set from general dataset %70 for training %30 for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Creating decision tree model object
clf = DecisionTreeClassifier()

# Training the model
clf = clf.fit(X_train, y_train)

# Making predictions using test dataset
y_pred = clf.predict(X_test)

# Printing accuracy
print("Accuracy with no optimization: ", metrics.accuracy_score(y_test, y_pred))

# Visualizing the tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=feature_cols, class_names=clf.classes_, filled=True, rounded=True)
plt.show()

# Creating decision tree model with optimized version of "gini" parameter.
clf = DecisionTreeClassifier(criterion="gini")

# Training the model
clf = clf.fit(X_train, y_train)

# Making predictions using test dataset
y_pred = clf.predict(X_test)

# Printing accuracy
print("Accuracy with gini optimization method: ", metrics.accuracy_score(y_test, y_pred))

# Visualizing the tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=feature_cols, class_names=clf.classes_, filled=True, rounded=True)
plt.show()



