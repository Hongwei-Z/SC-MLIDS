import helper
from sklearn.ensemble import RandomForestClassifier
from cryptography.fernet import Fernet
import socket
import os
import time
import zlib
import joblib
from tqdm import tqdm
import logging
from typing import Tuple, Optional, Any


# Constants
NUM_CLIENTS = 3
MODELS_DIR = "./received_models"
BUFFER_SIZE = 16384
HEADER_BUFFER = 1024
GLOBAL_MODEL_FILENAME = "global_model.joblib"


class Server:
    # Class implementing server.

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        # Initialize the server.

        self.host, self.port = host or helper.get_host_port()[0], port or helper.get_host_port()[1]
        self.server_socket = None
        self.resource_monitor = helper.ResourceMonitor()

        # Create directory for received models
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def start(self) -> None:
        # Start resource monitoring
        self.resource_monitor.start()

        # Initialize server socket
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(NUM_CLIENTS)

            print(f'Server started on {self.host}:{self.port}')
            print('Server is waiting for connections ...\n' + "-" * 80)

            # Process connections from all clients
            self._process_client_connections()

            # Train global model
            self._train_global_model()

        except Exception as e:
            logging.error(f"Server error: {str(e)}")
        finally:
            # Close server socket
            if self.server_socket:
                self.server_socket.close()

            # Stop resource monitoring
            self.resource_monitor.stop()
            print(f"{'All Done':-^80}")

    def _process_client_connections(self) -> None:
        # Process connections from clients.

        for i in range(NUM_CLIENTS):
            try:
                connection, address = self.server_socket.accept()
                self._handle_client(connection, address)
                connection.close()
            except Exception as e:
                logging.error(f"Error processing client {i + 1}: {str(e)}")

        print(f"{'All Clients Have Been Processed':-^80}")

    def _handle_client(self, connection: socket.socket, address: Tuple[str, int]) -> None:
        # Handle a client connection, receiving and processing the client's model.

        try:
            # Receive salt (fixed 16 bytes)
            salt = connection.recv(16)
            if not salt or len(salt) != 16:
                raise ValueError("Invalid salt received")

            # Receive client info header
            received_data = b""
            while True:
                part = connection.recv(HEADER_BUFFER)
                if not part:
                    raise ConnectionError("Connection closed during header receipt!")
                received_data += part
                if b"\n" in part:
                    break

            # Parse client information
            text_part = received_data.split(b"\n", 1)[0]
            client_info = text_part.decode('utf-8')
            client_id, file_size, ratio = client_info.split(',')
            client_id, file_size = int(client_id), int(file_size)
            ratio = float(ratio)

            print(f"{'Connected to Client ' + str(client_id):-^80}")
            print(f'Address: {address}\n')

            # Generate decryption key
            key = helper.generate_key(ratio, salt)
            print(f"Client {client_id} model decryption information:")
            print(f"Key: {key}\nSeed: {ratio}\nSalt: {salt}\n")

            # Receive the model file
            model_data = self._receive_model_file(connection, client_id, file_size)

            # Process and save the model
            self._process_client_model(model_data, client_id, key)

            print(f"{'Client ' + str(client_id) + ' Completed':-^80}\n")

        except Exception as e:
            logging.error(f"Error handling client at {address}: {str(e)}")

    def _receive_model_file(self, connection: socket.socket, client_id: int, file_size: int) -> bytes:
        # Receive encrypted and compressed model file from client.

        compressed_model = b''
        received_size = 0

        # Use progress bar to show file transfer progress
        with tqdm(total=file_size, desc=f"Receiving Client {client_id} model file",
                  unit="B", unit_scale=True, unit_divisor=1024) as progress:
            recv_start = time.time()

            while received_size < file_size:
                try:
                    # Calculate remaining bytes to receive
                    remaining = file_size - received_size

                    # Adjust buffer size for the final chunk if needed
                    buffer_size = min(BUFFER_SIZE, remaining)

                    data = connection.recv(buffer_size)
                    if not data:
                        break

                    compressed_model += data
                    received_size += len(data)
                    progress.update(len(data))

                except socket.timeout:
                    logging.warning(f"Socket timeout while receiving from client {client_id}!")
                    continue
                except Exception as e:
                    raise Exception(f"Error receiving data: {str(e)}!")

            recv_end = time.time()

        # Verify received file size
        self._verify_file_size(received_size, file_size, client_id, recv_end - recv_start)

        return compressed_model

    def _verify_file_size(self, received_size: int, expected_size: int,
                          client_id: int, transfer_time: float) -> None:
        # Verify that the received file size matches the expected size.

        print(f"Client {client_id} model file has been received, time spent: {transfer_time:.4f}s.")

        if received_size == expected_size:
            size_in_mb = expected_size / (1024 ** 2)
            print(f"Client {client_id} model file size: {size_in_mb:.4f}MB.")
        else:
            error_msg = f"Received file size {received_size} does not match expected size {expected_size}!"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _process_client_model(self, compressed_model: bytes, client_id: int, key: bytes) -> None:
        # Decompress, decrypt and save the client model.

        try:
            # Measure decompression and decryption time
            decr_start = time.time()

            # Decompress the model
            decompressed_model = zlib.decompress(compressed_model)

            # Decrypt the model
            cipher = Fernet(key)
            decrypted_model = cipher.decrypt(decompressed_model)

            decr_end = time.time()
            print(f"Client {client_id} model file has been decompressed and decrypted, "
                  f"time spent: {decr_end - decr_start:.4f}s.")

            # Save the model file
            filename = os.path.join(MODELS_DIR, f"client_{client_id}.joblib")
            with open(filename, 'wb') as file:
                file.write(decrypted_model)

            print(f"Client {client_id} model file has been saved to: {filename}.")

        except zlib.error:
            logging.error(f"Decompression error for client {client_id} model")
            raise
        except Exception as e:
            logging.error(f"Error processing client {client_id} model: {str(e)}")
            raise

    def _train_global_model(self) -> None:
        # Train the global model using network training data.

        print("\nGlobal model training ...")

        try:
            # Load the network training set
            X_train, y_train = helper.load_network_train_set()
            helper.print_label_distribution(y_train)

            # Train the network model
            network_model = RandomForestClassifier(random_state=42)

            train_start = time.time()
            network_model.fit(X_train, y_train)
            train_end = time.time()

            training_time = train_end - train_start
            print(f"Global model trained in {training_time:.4f}s.")

            # Save the global model
            global_file = os.path.join(MODELS_DIR, GLOBAL_MODEL_FILENAME)
            joblib.dump(network_model, filename=global_file, compress=3)
            print(f"Global model saved to: {global_file}.")

        except Exception as e:
            logging.error(f"Error training global model: {str(e)}")
            raise


def main():
    # Create and start server
    server = Server()
    server.start()


if __name__ == "__main__":
    main()
