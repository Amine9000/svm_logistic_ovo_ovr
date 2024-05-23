import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(max_iter=200)

ovo_classifier = OneVsOneClassifier(logreg)

ovo_classifier.fit(X_train, y_train)

y_pred = ovo_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"One-vs-One Logistic Regression Accuracy: {accuracy:.2f}")
