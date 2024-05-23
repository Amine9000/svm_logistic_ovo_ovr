import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

from utils import imshow
from logistic_one_rest import train_models as logistic_model_train
from logistic_one_rest import model_predict as logistic_model_predict
from svm_one_rest import train_models as SVM_model_train
from svm_one_rest import model_predict as SVM_model_predict

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


def ROC(y_test, y_pred, axes, row, col, label: str = ""):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    axes[row, col].plot(fpr, tpr, label=f'{label} = {auc:.2f}')
    axes[row, col].plot([0, 1], [0, 1], 'k--')
    axes[row, col].set_xlabel('False Positive Rate')
    axes[row, col].set_ylabel('True Positive Rate')
    axes[row, col].legend(loc='lower right')
    return auc


# train models --------------------------------------

if __name__ == '__main__':
    print("\n+----------------------------------------------------------+\n")
    logistic_models_info = logistic_model_train()
    logistic_predictions = logistic_model_predict(X_test, logistic_models_info)
    logistic_total_acc = accuracy_score(y_test, logistic_predictions)

    print()
    print(f"Logistic Regression Total accuracy : {logistic_total_acc:.3f}")
    print()
    print(y_test[:5])
    print(logistic_predictions[:5])
    print("\n+----------------------------------------------------------+\n")

    print("\n+----------------------------------------------------------+\n")
    SVM_models_info = SVM_model_train()
    SVM_predictions = SVM_model_predict(X_test, SVM_models_info)
    SVM_total_acc = accuracy_score(y_test, SVM_predictions)
    print()
    print(f"SVM Total accuracy :  {SVM_total_acc:.3f}")
    print()
    print(y_test[:5])
    print(SVM_predictions[:5])
    print("\n+----------------------------------------------------------+\n")

    num_classes = 4
    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    svm_logistic = 0

    for i in range(num_classes):
        ROC(np.where(y_test == i, 1, -1),
            np.where(logistic_predictions == i, 1, -1), axes, i//cols, i % cols, label="Logistic ROC AUC")
        ROC(np.where(y_test == i, 1, -1),
            np.where(SVM_predictions == i, 1, -1), axes, i//cols, i % cols, label="SVM ROC AUC")

    plt.show()
