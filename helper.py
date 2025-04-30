import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
import time
import psutil
from codecarbon import EmissionsTracker
import logging
from typing import Tuple, List, Dict, Union, Optional


# Constants
NUM_CLIENTS = 3
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8080
DEFAULT_TEST_SIZE = 0.2
SENSOR_FEATURE_COUNT = 3
DATASET_PATH = './datasets/merged_data.csv'
KDF_ITERATIONS = 100000
KDF_LENGTH = 32


def get_host_port() -> Tuple[str, int]:
    # Return the default host and port for network connections.
    return DEFAULT_HOST, DEFAULT_PORT


def split_dataset(test_size: float = DEFAULT_TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split the dataset into training and test sets.
    try:
        df = pd.read_csv(DATASET_PATH)
        train_set, test_set = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
        return train_set, test_set

    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}!")
    except Exception as e:
        raise Exception(f"Error splitting dataset: {str(e)}!")


def split_train_set() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split the training set into sensor and network datasets.
    train_set, _ = split_dataset()

    # Sensor dataset, first 3 features + labels
    sensor_train = train_set.iloc[:, list(range(SENSOR_FEATURE_COUNT)) + [-1]]

    # Network dataset (features 3-16)
    network_train = train_set.iloc[:, SENSOR_FEATURE_COUNT:17]

    return sensor_train, network_train


def load_test_set() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load and split the test set into sensor and network components.
    _, test_set = split_dataset()
    sensor_test = test_set.iloc[:, list(range(SENSOR_FEATURE_COUNT)) + [-1]]
    network_test = test_set.iloc[:, SENSOR_FEATURE_COUNT:17]
    return sensor_test, network_test


def load_sensor_train_set(client_id: int) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    # Load a portion of the sensor training set for a specific client.
    # This function distributes data evenly among NUM_CLIENTS clients.

    if not 1 <= client_id <= NUM_CLIENTS:
        logging.error(f"Error: The client number {client_id} exceeds the default of {NUM_CLIENTS}!")
        return None

    sensor_train, _ = split_train_set()
    x = sensor_train.iloc[:, :-1]
    y = sensor_train.iloc[:, -1]

    # Remove remainders to ensure even distribution
    remainder_count = len(x) % NUM_CLIENTS
    if remainder_count > 0:
        random_choose = np.random.choice(x.index, remainder_count, replace=False)
        x = x.drop(random_choose)
        y = y.drop(random_choose)

    # Split the dataset for clients (adjust index because client_id is 1-based)
    client_idx = client_id - 1
    split_indices = np.array_split(x.index, NUM_CLIENTS)
    client_indices = split_indices[client_idx]

    return x.loc[client_indices], y.loc[client_indices]


def load_network_train_set() -> Tuple[pd.DataFrame, pd.Series]:
    # Load the network training data.
    _, network_train = split_train_set()
    x_train = network_train.iloc[:, :-1]
    y_train = network_train.iloc[:, -1]
    return x_train, y_train


def get_label_ratio(y_train: pd.Series) -> float:
    # Compute the proportion of instances with label value 0.
    ratio = round(np.sum(y_train == 0) / len(y_train), 8)
    return ratio


def print_label_distribution(y: Union[pd.Series, np.ndarray]) -> Dict[int, int]:
    # Calculate and print the distribution of labels.
    unique, counts = np.unique(y, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    print("Label distribution:", label_counts, '\n')
    return label_counts


def get_metrics(y_test: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray],
                printout: bool = False) -> Tuple[float, float, float, float]:
    # Calculate classification metrics and optionally print them.

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    if printout:
        line = "*" * 19
        print(line)
        print(f"Accuracy : {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall   : {recall:.6f}")
        print(f"F1 Score : {f1:.6f}")
        print(line + '\n')

    return accuracy, precision, recall, f1


def generate_key(ratio: float, salt: bytes) -> bytes:
    # Generate a cryptographic key using a target ratio and salt.
    seed = str(ratio).encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KDF_LENGTH,
        salt=salt,
        iterations=KDF_ITERATIONS,
    )
    key = kdf.derive(seed)
    return base64.urlsafe_b64encode(key)


class ResourceMonitor:
    # Class for monitoring and reporting system resource usage.

    def __init__(self):
        # Initialize the resource monitor.
        logging.basicConfig(level=logging.ERROR)
        self.tracker = None
        self.start_time = None

    def start(self) -> None:
        # Start monitoring resources.

        self.tracker = EmissionsTracker(save_to_file=False, allow_multiple_runs=True)
        self.tracker.start()
        self.start_time = time.time()

        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        print(f"\nStart - CPU Usage: {cpu_usage}%")
        print(f"Start - Memory Usage: {memory_usage}%")

    def stop(self) -> Dict[str, float]:
        # Stop monitoring and return resource usage statistics.

        if not self.tracker or not self.start_time:
            raise RuntimeError("Resource monitoring was not started")

        self.tracker.stop()
        end_time = time.time()

        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        energy_consumed_kwh = self.tracker.final_emissions_data.energy_consumed
        co2_emission_kg = self.tracker.final_emissions_data.emissions
        execution_time = end_time - self.start_time

        # Print resource usage information
        line = "*" * 29
        print('\n' + line)
        print("Resource Usage:")
        print(f"Execution Time: {execution_time:.6f}s")
        print(f"Energy Consumed: {energy_consumed_kwh:.6f}kWh")
        print(f"CO₂ Emission: {co2_emission_kg:.6f}kg")
        print(f"End - CPU Usage: {cpu_usage}%")
        print(f"End - Memory Usage: {memory_usage}%")
        print(line + '\n')

        # Return metrics as a dictionary
        return {
            "execution_time": execution_time,
            "energy_consumed_kwh": energy_consumed_kwh,
            "co2_emission_kg": co2_emission_kg,
            "final_cpu_usage": cpu_usage,
            "final_memory_usage": memory_usage
        }


# Legacy function wrappers for backward compatibility
def monitor_resources():
    # Legacy function for starting resource monitoring.
    monitor = ResourceMonitor()
    monitor.start()
    return monitor.tracker, monitor.start_time


def stop_monitoring(tracker, start_time):
    # Legacy function for stopping resource monitoring.
    temp_monitor = ResourceMonitor()
    temp_monitor.tracker = tracker
    temp_monitor.start_time = start_time
    return temp_monitor.stop()
