import helper
from sklearn.ensemble import RandomForestClassifier
from cryptography.fernet import Fernet
import socket
import joblib
import zlib
import os
import time
import logging
from typing import Optional, Tuple, Union, Dict, Any
import argparse


# Constants
CLIENT_MODELS_DIR = "./client_models"
SOCKET_TIMEOUT = 30
CHUNK_SIZE = 16384
COMPRESSION_LEVEL = 9


class Client:
    # Class implementing clients

    def __init__(self, client_id: Optional[int] = None,
                 host: Optional[str] = None, port: Optional[int] = None):
        # Initialize the client.

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Initialize client ID (prompt if not provided)
        self.client_id = client_id or self._get_client_id_from_user()

        # Initialize server connection details
        self.host, self.port = host or helper.get_host_port()[0], port or helper.get_host_port()[1]

        # Initialize resource monitoring
        self.resource_monitor = helper.ResourceMonitor()

        # Create directory for client models
        os.makedirs(CLIENT_MODELS_DIR, exist_ok=True)

        # Initialize model attributes
        self.model = None
        self.ratio = None
        self.salt = None
        self.key = None
        self.model_path = os.path.join(CLIENT_MODELS_DIR, f"client_{self.client_id}.joblib")

        # Print client header
        print(f"{'Client ' + str(self.client_id):-^80}")

    def _get_client_id_from_user(self) -> int:
        # Prompt the user for a client ID.
        client_id = input("Enter Client ID (int): ")
        while not client_id.isdigit():
            client_id = input("Invalid ID, re-enter: ")
        return int(client_id)

    def train_model(self) -> None:
        # Train the local model using sensor data specific to this client.
        print("Loading sensor data and training local model...")

        try:
            # Load data for this client
            X_train, y_train = helper.load_sensor_train_set(self.client_id)
            if X_train is None or y_train is None:
                raise ValueError(f"Failed to load data for client {self.client_id}")

            # Print label distribution
            helper.print_label_distribution(y_train)

            # Calculate label ratio (used for encryption)
            self.ratio = helper.get_label_ratio(y_train)

            # Initialize and train model
            self.model = RandomForestClassifier(random_state=42)

            # Train the model with timing
            train_start = time.time()
            self.model.fit(X_train, y_train)
            train_end = time.time()
            train_time = train_end - train_start

            print(f"Client {self.client_id} model trained in {train_time:.4f}s.")

            # Save the model
            self._save_model()

        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def _save_model(self) -> None:
        # Save the trained model to a file.
        try:
            # Save model with compression
            joblib.dump(self.model, filename=self.model_path, compress=3)
            print(f"Client {self.client_id} model saved to: {self.model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def prepare_encryption(self) -> None:
        # Generate encryption key and salt for secure model transmission.

        try:
            # Generate random salt
            self.salt = os.urandom(16)

            # Generate encryption key based on model properties
            self.key = helper.generate_key(self.ratio, self.salt)

            print(f"\nClient {self.client_id} model encryption information:")
            print(f"Key: {self.key}\nSeed: {self.ratio}\nSalt: {self.salt}\n")

        except Exception as e:
            logging.error(f"Error preparing encryption: {str(e)}")
            raise

    def encrypt_and_compress_model(self) -> bytes:
        # Encrypt and compress the model for transmission.

        try:
            # Read model file
            with open(self.model_path, 'rb') as file:
                model_data = file.read()

            original_size = len(model_data)
            print(f"Model file size: {(original_size / (1024 ** 2)):.4f}MB")

            # Encrypt the model
            encr_start = time.time()
            cipher = Fernet(self.key)
            encrypted_model = cipher.encrypt(model_data)
            encr_end = time.time()

            encrypted_size = len(encrypted_model)
            print(f"Model file encrypted, file size: {(encrypted_size / (1024 ** 2)):.4f}MB, "
                  f"time spent: {encr_end - encr_start:.4f}s.")

            # Compress the encrypted model
            comp_start = time.time()
            compressed_model = zlib.compress(encrypted_model, level=COMPRESSION_LEVEL)
            comp_end = time.time()

            compressed_size = len(compressed_model)
            print(f"Model file compressed, file size: {(compressed_size / (1024 ** 2)):.4f}MB, "
                  f"time spent: {comp_end - comp_start:.4f}s.")
            print(f"Compression ratio: {original_size / compressed_size:.2f}x")

            return compressed_model

        except Exception as e:
            logging.error(f"Error encrypting and compressing model: {str(e)}")
            raise

    def send_model_to_server(self, compressed_model: bytes) -> None:
        # Send the encrypted and compressed model to the server.
        client_socket = None

        try:
            # Create socket and set timeout
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(SOCKET_TIMEOUT)

            # Connect to server
            client_socket.connect((self.host, self.port))
            print(f"\nClient {self.client_id} has connected to the server at {self.host}:{self.port}.")

            # Send salt (16 bytes)
            client_socket.sendall(self.salt)

            # Send metadata (client_id, file_size, ratio)
            file_size = len(compressed_model)
            metadata = f'{self.client_id},{file_size},{self.ratio}\n'.encode('utf-8')
            client_socket.sendall(metadata)

            # Send compressed and encrypted model
            send_start = time.time()

            # Send in chunks to handle large files
            bytes_sent = 0
            while bytes_sent < file_size:
                # Calculate the chunk to send
                remaining = file_size - bytes_sent
                chunk_size = min(CHUNK_SIZE, remaining)
                chunk = compressed_model[bytes_sent:bytes_sent + chunk_size]

                # Send the chunk
                sent = client_socket.send(chunk)
                if sent == 0:
                    raise RuntimeError("Socket connection broken")

                bytes_sent += sent

            send_end = time.time()

            print(f"Model file has been sent from Client {self.client_id}, "
                  f"file size: {(file_size / (1024 ** 2)):.4f}MB, "
                  f"time spent: {send_end - send_start:.4f}s.")

        except socket.timeout:
            logging.error("Connection timed out while sending model to server!")
            raise
        except ConnectionRefusedError:
            logging.error(f"Connection refused by server at {self.host}:{self.port}!")
            raise
        except Exception as e:
            logging.error(f"Error sending model to server: {str(e)}!")
            raise
        finally:
            if client_socket:
                client_socket.close()

    def run(self) -> None:
        # Execute the complete client workflow.

        try:
            # Start resource monitoring
            self.resource_monitor.start()

            # Train the local model
            self.train_model()

            # Prepare encryption materials
            self.prepare_encryption()

            # Encrypt and compress the model
            compressed_model = self.encrypt_and_compress_model()

            # Send model to server
            self.send_model_to_server(compressed_model)

        except Exception as e:
            logging.error(f"Client {self.client_id} encountered an error: {str(e)}!")
        finally:
            # Stop resource monitoring
            self.resource_monitor.stop()
            print(f"{'Client ' + str(self.client_id) + ' Completed':-^80}")


def parse_arguments():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Client')
    parser.add_argument('--id', type=int, help='Client ID')
    parser.add_argument('--host', type=str, help='Server hostname/IP')
    parser.add_argument('--port', type=int, help='Server port')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Create and run client
    client = Client(client_id=args.id, host=args.host, port=args.port)
    client.run()


if __name__ == "__main__":
    main()
