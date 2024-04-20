import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split

col_names = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year', 'Brand']
data = pd.read_csv("../../Downloads/cars.csv", names=col_names, header=None)

# Normalize the 'Brand' column by stripping whitespaces
data['Brand'] = data['Brand'].str.strip()

# Extract features and target variable
feature_cols = ['MPG', 'Cylinders', 'CubicInches', 'Horse Power', 'Weight', 'Time to 60 MPH', 'Year']
X = data[feature_cols]
y = data['Brand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions, zero_division='warn', average='weighted')
recall = recall_score(y_test, predictions, zero_division='warn', average='weighted')
f1 = f1_score(y_test, predictions, zero_division='warn', average='weighted')

# Data visualization


metrics_df = pd.DataFrame({
    'Class': model.classes_,
    'Precision': precision_score(y_test, predictions, average=None),
    'Recall': recall_score(y_test, predictions, average=None),
    'F1-Score': f1_score(y_test, predictions, average=None)
})

# Plot model evaluation metrics
plt.figure(figsize=(10, 6))
metrics_df.set_index('Class').plot(kind='bar')
plt.title('Model Evaluation Metrics by Class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.show()

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Brand', data=data)
plt.title('Class Distribution')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature Importances
feature_importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


'''Accuracy is the overall correctness of predictions. Precision is the accuracy of positive predictions. Recall is 
the ability to correctly identify positive instances. F1-Score is the balance between precision and recall. 
Visualizing these metrics using a bar plot to understand the model's performance for each class. Visualizing the 
confusion matrix helps to understand the distribution of predicted and actual class labels.'''