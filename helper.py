import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# TODO: 在全局模型中增加平均计算，按照子模型的训练集比例，准确度，计算百分比，三个模型乘百分比，计算最终分类结果


# Keep 10% of dataset for global model
def load_testset(test_size: float):
    df = pd.read_csv('./datasets/label_data.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    trainset = pd.concat([X_train, y_train], axis=1)
    return trainset, X_test, y_test


# Use rest of 90% dataset
def load_dataset(client_id: int):
    df, _, _ = load_testset(test_size=0.1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the dataset evenly into thirds, removing the remainders
    np.random.seed(42)
    random_choose = np.random.choice(X.index, (len(X) % 3), replace=False)
    X = X.drop(random_choose)
    y = y.drop(random_choose)

    # Split the dataset into 3 subsets for 3 clients
    X_split, y_split = np.split(X, 3), np.split(y, 3)
    X1, y1 = X_split[0], y_split[0]
    X2, y2 = X_split[1], y_split[1]
    X3, y3 = X_split[2], y_split[2]

    # Split the training set and testing set in 80% ratio
    X_train, y_train, X_test, y_test = [], [], [], []
    train_size = 0.8

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1,train_size=train_size, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,train_size=train_size, random_state=42)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3,train_size=train_size, random_state=42)

    X_train.append(X1_train)
    X_train.append(X2_train)
    X_train.append(X3_train)

    y_train.append(y1_train)
    y_train.append(y2_train)
    y_train.append(y3_train)

    X_test.append(X1_test)
    X_test.append(X2_test)
    X_test.append(X3_test)

    y_test.append(y1_test)
    y_test.append(y2_test)
    y_test.append(y3_test)

    # Each of the following is divided equally into thirds
    return X_train[client_id], y_train[client_id], X_test[client_id], y_test[client_id]


# Print the label distribution
def label_distribution(y_train, y_test):
    unique, counts = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts))
    print("Label distribution in the training set:", train_counts)
    unique, counts = np.unique(y_test, return_counts=True)
    test_counts = dict(zip(unique, counts))
    print("Label distribution in the testing set:", test_counts, '\n')


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
    print(line)
