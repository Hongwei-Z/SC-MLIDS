import helper
import socket
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import time
import warnings
warnings.simplefilter('ignore')


client_id = 1
print(f"Client {client_id}:\n")
HOST = '127.0.0.1'
PORT = 8080

# Get the dataset for local model
X_train, y_train = helper.load_trainset(client_id - 1)
helper.label_distribution(y_train)
ratio = helper.label_ratio(y_train)

# Create and train the local model
model = RandomForestClassifier()
train_start = time.time()
model.fit(X_train, y_train)
train_end = time.time()
train_time = train_end - train_start
print(f"Client {client_id} model training completed in {train_time:.6f} seconds.")

# Generate model file
filename = f'local_model_{client_id}.joblib'
joblib.dump(model, filename=filename)
print(f"Client {client_id} model saved locally.")

# Get the size of the file
file_size = os.path.getsize(filename)


# Sending the client ID, file size, label ratio, and model file to the server
try:
    client = socket.socket()
    client.connect((HOST, PORT))
    print(f"Client {client_id} has connected to the server.")

    client.sendall(f'{client_id},{file_size},{ratio}'.encode())
    print(f"Client ID has been sent from Client {client_id}.")
    print(f"Model file size has been sent from Client {client_id}.")

    start_time = time.time()
    with open(filename, 'rb') as file:
        sendfile = file.read()
    client.sendall(sendfile)
    end_time = time.time()

    print(f"Model file has been sent from Client {client_id}.")
    print(f"Time taken to send model file: {end_time - start_time:.6f} seconds.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()
