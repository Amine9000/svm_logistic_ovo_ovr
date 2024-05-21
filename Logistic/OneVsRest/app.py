import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from utils import imshow


data = load_digits()

X, y = data.data, data.target.reshape(-1, 1)


# imshow(X[:10], y[:10], suptitle="Images from breast cancer dataset")

# One vs Rest ------------------------------

# spliting ------------------------------

X_train_or, X_test_or, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# scaling ------------------------------

scaler = StandardScaler()

y_train = y_train.ravel()
y_test = y_test.ravel()

X_train = scaler.fit_transform(X_train_or)
X_test = scaler.transform(X_test_or)


def train_models():
    models = []

    for digit in range(10):
        d_y_train = np.where(y_train == digit, 1, -1)
        d_y_test = np.where(y_test == digit, 1, -1)
        model = LogisticRegression()
        model.fit(X_train, d_y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(d_y_test, predictions)
        print("Accuracy for digit ", digit, " is ", acc)
        models.append(model)

    return models


def model_predict(x):
    predictions = []
    for model in models:
        prediction = model.predict(x)
        predictions.append(prediction)
    arr = np.array(predictions)
    return np.argmax(arr, axis=0)


models = train_models()
predictions = model_predict(X_test)
total_acc = accuracy_score(y_test, predictions)
print("Total accuracy is ", total_acc)
print(y_test[:5])
print(predictions[:5])

imshow(X_test_or[:10], predictions[:10],
       suptitle="Images from digits dataset predictions")