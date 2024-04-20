import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
import seaborn as sns
import warnings

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')

# Read CSV File
col_names = ['MPG', 'Cylinders', 'CubicInches', 'HorsePower', 'Weight', 'Time to 60 MPH', 'Year', 'Brand']
data = 'cars.csv'
dataFrame = pd.read_csv(data, header=None, names=col_names)

# Choose target,split data in features
X = dataFrame.drop(['Brand'], axis=1)
y = dataFrame['Brand']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Prediction without any parameter
randomForest = RandomForestClassifier()
randomForest.fit(X_train, y_train)
randomForest_prediction = randomForest.predict(X_test)
print('Accuracy without parameters:', accuracy_score(y_test, randomForest_prediction))

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {'n_estimators': randint(150, 200), 'max_depth': randint(10, 15)}
rand_search = RandomizedSearchCV(randomForest, param_distributions=param_dist, cv=5)
rand_search.fit(X_train, y_train)
tunedRandomForest = rand_search.best_estimator_
print('Best parameters:', rand_search.best_params_)

# Prediction with hyperparameter tuning
tunedForest_prediction = tunedRandomForest.predict(X_test)
accuracy = accuracy_score(y_test, tunedForest_prediction)
print('Accuracy:', accuracy)

# Calculate precision and recall
precision = precision_score(y_test, tunedForest_prediction, average='macro', pos_label='positive')
recall = recall_score(y_test, tunedForest_prediction, average='macro', pos_label='positive')
print('Precision:', precision)
print('Recall:', recall)

# Display confusion matrix
cm = confusion_matrix(y_test, tunedForest_prediction)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=randomForest.classes_, yticklabels=randomForest.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Find and display the biggest predictor in data.
feature_importances = pd.Series(tunedRandomForest.feature_importances_, index=X_train.columns)
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


