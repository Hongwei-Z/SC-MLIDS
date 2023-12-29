import helper
from sklearn.ensemble import RandomForestClassifier
from cryptography.fernet import Fernet
import socket
import os
import time
import zlib
import joblib
from tqdm import tqdm


CLIENTS = 3


# Receive model info and file from clients
def connect_clients(connection, address):
    path = "./received_models"
    os.makedirs(path, exist_ok=True)

    # Receive salt, client id, file size, ratio
    salt = connection.recv(16)

    received_data = b""
    while True:
        part = connection.recv(1024)
        received_data += part
        if b"\n" in part:
            break
    text_part = received_data.split(b"\n", 1)[0]
    client_info = text_part.decode('utf-8')
    client_id, file_size, ratio = client_info.split(',')
    print(f'Connected to Client {client_id}:\nAddress: {address}\n')

    # Generate the key
    key = helper.generate_key(float(ratio), salt)
    print(f"Client {client_id} model decryption information:")
    print(f"Key: {key}\nSeed: {ratio}\nSalt: {salt}\n")

    # Receive the compressed and encrypted model file
    compressed_model = b''
    received_size = 0
    file_size = int(file_size)

    with tqdm(range(file_size), f"Receiving Client {client_id} model file", unit="B",
              unit_scale=True, unit_divisor=1024) as progress:
        recv_start = time.time()
        while received_size < file_size:
            data = connection.recv(16384)
            if not data:
                break
            compressed_model += data
            received_size += len(data)
            progress.update(len(data))
        progress.close()
        recv_end = time.time()
        print(f"Client {client_id} model file has been received, time spent: {recv_end - recv_start:.4f} seconds")

    # Match file size
    if received_size == file_size:
        size_in_mb = file_size / (1024 ** 2)
        print(f"Client {client_id} model file size: {size_in_mb:.4f} MB")
    else:
        print(f"Error: Received file size {received_size} does not match expected size {file_size}")

    # Decompress and decrypt model file
    decr_start = time.time()
    decompressed_model = zlib.decompress(compressed_model)
    decrypted_model = Fernet(key).decrypt(decompressed_model)
    decr_end = time.time()
    print(f"Client {client_id} model file has been decompressed and decrypted, "
          f"time spent: {decr_end - decr_start:.4f} seconds")

    # Save the model file
    filename = f"{path}/client_{client_id}.joblib"
    with open(filename, 'wb') as file:
        file.write(decrypted_model)
    print(f"Client {client_id} model file has been saved to: {filename}")
    print("-" * 45 + f" Client {client_id} Completed " + "-" * 45 + "\n")


server = socket.socket()
host, port = helper.get_host_port()
server.bind((host, port))
server.listen(CLIENTS)

# Process all clients
print('Server is waiting for connections ...\n' + "-" * 110)
for _ in range(CLIENTS):
    connection, address = server.accept()
    connect_clients(connection, address)
    connection.close()
print("-" * 39 + ' All Clients Have Been Processed ' + "-" * 38)

# Load the network training set
print("\nTraining the global model using network traffic data ...")
X_train, y_train = helper.load_network_train_set()
helper.print_label_distribution(y_train)

# Train the network model
network_model = RandomForestClassifier()
train_start = time.time()
network_model.fit(X_train, y_train)
train_end = time.time()
print(f"Global model training completed in {(train_end - train_start):.4f} seconds.")

# Generate model file
global_file = './received_models/global_model.joblib'
joblib.dump(network_model, filename=global_file)
print(f"Global model saved to: {global_file}")

print("-" * 50 + ' All Done ' + "-" * 50)
