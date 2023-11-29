import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64


# Getting the testing set from a dataset
def load_testset(test_size: float):
    df = pd.read_csv('./datasets/label_data.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    trainset = pd.concat([X_train, y_train], axis=1)
    return trainset, X_test, y_test


# Divide the training set into thirds for three clients
def load_trainset(client_id: int):
    df, _, _ = load_testset(test_size=0.3)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the dataset evenly into thirds, removing the remainders
    random_choose = np.random.choice(X.index, (len(X) % 3), replace=False)
    X = X.drop(random_choose)
    y = y.drop(random_choose)

    # Split the dataset into 3 subsets for 3 clients
    X_train, y_train = np.split(X, 3), np.split(y, 3)
    return X_train[client_id], y_train[client_id]


# Compute the proportion of 0
def label_ratio(y_train) -> float:
    ratio = round(np.sum(y_train == 0) / len(y_train), 8)
    return ratio


# Print the label distribution
def label_distribution(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts))
    print("Label distribution in the training set:", train_counts, '\n')


# Print metrics
def display_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    line = "-" * 21
    print(line)
    print(f"Accuracy : {accuracy:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall   : {recall:.8f}")
    print(f"F1 Score : {f1:.8f}")
    print(line + '\n\n')


# Generate key using label_ratio
def generate_key(ratio, salt):
    seed = str(ratio).encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(seed)
    return base64.urlsafe_b64encode(key)
