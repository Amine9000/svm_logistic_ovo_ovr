import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from utils import imshow


data = load_digits()

X, y = data.data, data.target.reshape(-1, 1)


# One vs One ------------------------------

# spliting ------------------------------

X_train_or, X_test_or, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# scaling ------------------------------

scaler = StandardScaler()

y_train = y_train.ravel()
y_test = y_test.ravel()

X_train = scaler.fit_transform(X_train_or)
X_test = scaler.transform(X_test_or)


# train models --------------------------------------

num_classes = 10


def train_models():
    models = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            # selecting to classes
            x_data_train = X_train[(y_train == i) | (y_train == j)]
            x_data_test = X_test[(y_test == i) | (y_test == j)]
            y_data_train = y_train[(y_train == i) | (y_train == j)]
            y_data_test = y_test[(y_test == i) | (y_test == j)]

            # changing the labels
            # y_data_train = np.where(y_data_train == i, 1, -1)
            # y_data_test = np.where(y_data_test == i, 1, -1)

            # model
            model = SVC()
            model.fit(x_data_train, y_data_train)
            y_pred = model.predict(x_data_test)
            accuracy = accuracy_score(y_data_test, y_pred)
            models.append([model, accuracy])

    return models


def model_predict(x, models):
    predictions = []
    for model in models:
        predictions.append(model.predict(x))
    predictions = np.array(predictions).T
    predictions = np.array([Counter(el).most_common(1)[0][0]
                            for el in predictions])
    return predictions


if __name__ == '__main__':
    models_info = train_models()
    models = [model[0] for model in models_info]
    accuracies = [model[1] for model in models_info]

    predictions = model_predict(X_test, models)
    accuracies_total = accuracy_score(y_test, predictions)
    print("accuracy total is : ", accuracies_total)
    print(y_test[:10])
    print(predictions[:10])

    imshow(X_test_or[:10], predictions[:10],
           suptitle="Images from digits dataset predictions")
