import socket
import threading
import os


print(f"Server:\n")
HOST = "127.0.0.1"
PORT = 8080


def handle_clients(connection, address):
    path = "./models"
    os.makedirs(path, exist_ok=True)

    # Receive client ID
    client_id = connection.recv(1024).decode()

    filename = f"{path}/model_client_{client_id}.joblib"
    print(f'Connected to Client {client_id}\nAddress: {address}\n')

    with connection, open(filename, 'wb') as file:
        while True:
            receive_file = connection.recv(4096)
            if not receive_file:
                print(f"Model file from {client_id} not received.")
                break
            file.write(receive_file)
    print(f"Model file has been received from Client {client_id} and saved on {filename}.")


if __name__ == '__main__':
    server = socket.socket()
    server.bind((HOST, PORT))
    server.listen(3)  # 3 clients

    print('Server is waiting for connections...')
    while True:
        client, address = server.accept()
        client_thread = threading.Thread(target=handle_clients, args=(client, address))
        client_thread.start()
