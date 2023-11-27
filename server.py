import socket
import threading
import os
import time


print(f"Server:\n")
HOST = "127.0.0.1"
PORT = 8080


# Receive model info and file from clients
def connect_clients(connection, address):
    path = "./models"
    os.makedirs(path, exist_ok=True)

    # Receive client ID and file size
    id_size = connection.recv(1024).decode()
    client_id, file_size, ratio = id_size.split(',')

    # Store the ratio in a file
    with open('./models/ratio.txt', 'a') as f:
        f.write(f"{client_id}:{ratio}\n")

    print(f'Connected to Client {client_id}\nAddress: {address}\n')

    start_time = time.time()

    filename = f"{path}/model_client_{client_id}.joblib"
    received_size = 0
    with connection, open(filename, 'wb') as file:
        while received_size < int(file_size):
            data = connection.recv(4096)
            if not data:
                break
            file.write(data)
            received_size += len(data)

    end_time = time.time()

    if received_size == int(file_size):
        print(f"Model file has been received from Client {client_id} and saved on {filename}.")
    else:
        print(f"Error: Received file size {received_size} does not match expected size {int(file_size)}.")

    print(f"Time taken to receive model file from Client {client_id}: {end_time - start_time:.6f} seconds.\n")


if __name__ == '__main__':
    server = socket.socket()
    server.bind((HOST, PORT))
    server.listen(3)  # 3 clients

    print('Server is waiting for connections...\n')
    while True:
        client, address = server.accept()
        client_thread = threading.Thread(target=connect_clients, args=(client, address))
        client_thread.start()
