## MCS Project
### A Novel Federated Learning-Based Distributed Machine Learning Security Scheme for Wireless Sensor Networks
#### Author: Hongwei Zhang

### Getting Started
1. This project contains following source code files:
    ```
    helper.py
    server.py
    client1.py
    client2.py
    client3.py
    ```
2. Install Required Libraries:
    ```bash
   pip install -r requirements.txt
   ```

### Running

Open two terminals, one for the server and another for the clients.
 1. In the first terminal, load the helper and start the server:
    ```bash
    python helper.py
    python server.py
    ```
 2. In the second terminal, run each of the three clients:
    ```bash
    # Note: Run the next client only after seeing the previous client's task completed in the server terminal.
    python client1.py
    python client2.py
    python client3.py
    ```


### Evaluating

1. **Server running results (first terminal):**
   ```bash
    C:\Users>python helper.py
    C:\Users>python server.py
    Server is waiting for connections...
    -------------------------------------------------------------------------------------
    Connected to Client 1
    Address: ('127.0.0.1', 53270)
    
    Client 1 model decryption information:
    Key: b'KFOYj6Wz5mRjRS8D_snzl7imnLVlaup-yCDnPsDVR8A='
    Seed: 0.096988
    Salt: b'`9H2p\xf8\xad\xcc\xcb|^\xa4^?E\xa5'
    
    Client 1 model file has been received, time spent: 366.9219 seconds
    Client 1 model file size: 231.2357 MB
    Client 1 model file has been decompressed and decrypted, time spent: 2.0524 seconds
    Client 1 model file has been saved to: ./received_models/client_1.joblib
    -------------------------------- Client 1 Completed --------------------------------
    
    Connected to Client 2
    Address: ('127.0.0.1', 53431)
    
    Client 2 model decryption information:
    Key: b'VzS17H_kgFFRqLlNWP2Z4UFzd4pw1HN24vaxf-hxMSw='
    Seed: 0.09754009
    Salt: b'\xc2\xfc\xd5\xa3\x10b\xfc?\x9c\xc4\xd0\xa1\xedt\x8a\xb0'
    
    Client 2 model file has been received, time spent: 290.1457 seconds
    Client 2 model file size: 229.7702 MB
    Client 2 model file has been decompressed and decrypted, time spent: 1.8966 seconds
    Client 2 model file has been saved to: ./received_models/client_2.joblib
    -------------------------------- Client 2 Completed --------------------------------
    
    Connected to Client 3
    Address: ('127.0.0.1', 53455)
    
    Client 3 model decryption information:
    Key: b'43f0s_sy47ygEq8MLZTjZrLbuonz8-YPyIJ0lGzQeGI='
    Seed: 0.09719451
    Salt: b"\xca\xed\x9cmx'\xa0\xb2\x0bg\xaa\xd3x)\xafS"
    
    Client 3 model file has been received, time spent: 292.5300 seconds
    Client 3 model file size: 230.6720 MB
    Client 3 model file has been decompressed and decrypted, time spent: 1.9358 seconds
    Client 3 model file has been saved to: ./received_models/client_3.joblib
    -------------------------------- Client 3 Completed --------------------------------
    
    -------------------------- All Clients Have Been Processed --------------------------
    -------------------------------------------------------------------------------------
    Training the global model using network traffic data...
    Label distribution in the training set: {0: 211537, 1: 1967446}
    
    Global model training completed in 211.3654 seconds.
    Global model saved to: ./received_models/global_model.joblib
    ------------------------------------- All Done -------------------------------------
    C:\Users>
   ```
2. **Clients running results (second terminal):**
    ```bash
    C:\Users>python client1.py
    Client 1:
    
    Label distribution in the training set: {0: 70445, 1: 655882}
    
    Client 1 model training completed in 99.2599 seconds.
    Client 1 model saved to: ./client_models/client_1.joblib
    
    Client 1 model encryption information:
    Key: b'KFOYj6Wz5mRjRS8D_snzl7imnLVlaup-yCDnPsDVR8A='
    Seed: 0.096988
    Salt: b'`9H2p\xf8\xad\xcc\xcb|^\xa4^?E\xa5'
    
    Model file size: 228.9369 MB
    Model file encrypted, file size: 305.2493 MB, time spent: 1.7334 seconds
    Model file compressed, file size: 231.2357 MB, time spent: 12.3500 seconds
    
    Client 1 has connected to the server.
    Model file has been sent from Client 1, file size: 231.2357 MB, time spent: 0.0528 seconds
    
    C:\Users>python client2.py
    Client 2:
    
    Label distribution in the training set: {0: 70846, 1: 655481}
    
    Client 2 model training completed in 78.5733 seconds.
    Client 2 model saved to: ./client_models/client_2.joblib
    
    Client 2 model encryption information:
    Key: b'VzS17H_kgFFRqLlNWP2Z4UFzd4pw1HN24vaxf-hxMSw='
    Seed: 0.09754009
    Salt: b'\xc2\xfc\xd5\xa3\x10b\xfc?\x9c\xc4\xd0\xa1\xedt\x8a\xb0'
    
    Model file size: 227.4859 MB
    Model file encrypted, file size: 303.3146 MB, time spent: 1.2123 seconds
    Model file compressed, file size: 229.7702 MB, time spent: 8.1777 seconds
    
    Client 2 has connected to the server.
    Model file has been sent from Client 2, file size: 229.7702 MB, time spent: 0.0314 seconds
    
    C:\Users>python client3.py
    Client 3:
    
    Label distribution in the training set: {0: 70595, 1: 655732}
    
    Client 3 model training completed in 70.5190 seconds.
    Client 3 model saved to: ./client_models/client_3.joblib
    
    Client 3 model encryption information:
    Key: b'43f0s_sy47ygEq8MLZTjZrLbuonz8-YPyIJ0lGzQeGI='
    Seed: 0.09719451
    Salt: b"\xca\xed\x9cmx'\xa0\xb2\x0bg\xaa\xd3x)\xafS"
    
    Model file size: 228.3787 MB
    Model file encrypted, file size: 304.5050 MB, time spent: 1.2344 seconds
    Model file compressed, file size: 230.6720 MB, time spent: 8.2768 seconds
    
    Client 3 has connected to the server.
    Model file has been sent from Client 3, file size: 230.6720 MB, time spent: 0.0314 seconds
    
    C:\Users>
    ```



Citations:   
1. [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2. [Real Python: Socket Programming in Python](https://realpython.com/python-sockets/)
3. [Stack Overflow: Send big file over socket](https://stackoverflow.com/questions/56194446/send-big-file-over-socket)
4. [Cryptography: Fernet](https://cryptography.io/en/latest/fernet/)