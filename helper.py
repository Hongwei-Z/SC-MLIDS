import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64


# Split the train set and test set
def split_dataset(test_size=0.3):
    df = pd.read_csv('./datasets/merged_data.csv')
    train_set, test_set = train_test_split(df, test_size=test_size, shuffle=True)
    return train_set, test_set


# Split the train set to sensor train set and network train set
def split_train_set():
    train_set, _ = split_dataset()
    sensor_train = train_set.iloc[:, list(range(3)) + [-1]]
    network_train = train_set.iloc[:, 3:17]
    return sensor_train, network_train


# Return the test set
def load_test_set():
    _, test_set = split_dataset()
    return test_set


# Split the sensor train set evenly into thirds, return one set
def load_sensor_train_set(client_id: int):
    if 0 <= client_id <= 2:  # Default: 3 clients
        sensor_train, _ = split_train_set()
        X = sensor_train.iloc[:, :-1]
        y = sensor_train.iloc[:, -1]

        # Split the dataset evenly into thirds, removing the remainders
        random_choose = np.random.choice(X.index, (len(X) % 3), replace=False)
        X = X.drop(random_choose)
        y = y.drop(random_choose)

        # Split the dataset into 3 subsets for 3 clients
        X_train, y_train = np.split(X, 3), np.split(y, 3)
        return X_train[client_id], y_train[client_id]
    else:
        print("Error: The client number exceeds the default of 3. Please modify the code to accept more clients.")
        return


# Load network traffic data
def load_network_train_set():
    _, network_train = split_train_set()
    return network_train


# Compute the proportion of 0
def get_label_ratio(y_train) -> float:
    ratio = round(np.sum(y_train == 0) / len(y_train), 8)
    return ratio


# Print the label distribution
def print_label_distribution(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts))
    print("Label distribution in the training set:", train_counts, '\n')


# Print metrics
def get_metrics(y_test, y_pred, printout=False):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    if printout:
        line = "-" * 21
        print(line)
        print(f"Accuracy : {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall   : {recall:.8f}")
        print(f"F1 Score : {f1:.8f}")
        print(line + '\n')

    return accuracy, precision, recall, f1


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
