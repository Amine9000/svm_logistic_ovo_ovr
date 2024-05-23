import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(max_iter=200)

ovr_classifier = OneVsRestClassifier(logreg)

ovr_classifier.fit(X_train, y_train)

y_pred = ovr_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"One-vs-Rest Logistic Regression Accuracy: {accuracy:.2f}")
