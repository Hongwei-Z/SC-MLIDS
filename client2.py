import socket
import joblib
from sklearn.ensemble import RandomForestClassifier
import helper
import warnings
warnings.simplefilter('ignore')


client_id = 2
print(f"Client {client_id}:\n")
HOST = "127.0.0.1"
PORT = 8080

# Get the dataset for local model
X_train, y_train, X_test, y_test = helper.load_dataset(client_id - 1)

# Create and train the local model
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

helper.display_metrics(y_test, y_pred)


# Generate model file
joblib.dump(model, f'model{client_id}.joblib')

client = socket.socket()
client.connect((HOST, PORT))

# Sending the model file to the server
try:
    # Send client ID
    client.sendall(str(client_id).encode())

    with open(f'model{client_id}.joblib', 'rb') as file:
        sendfile = file.read()
    client.sendall(sendfile)
    print(f"Model file has been sent from Client {client_id}.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()
