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
    Address: ('127.0.0.1', 51105)
    
    Client 1 model decryption information:
    Key: b'msXoRJIMUvABzyvm7jtxKImOLwbCNANq8IUrpbMhkWE='
    Seed: 0.86087805
    Salt: b'+\xd6\xdc\xeb\xaa\xf7R\x86\xf5wA\x08\x0f\xcd\xda<'
    
    Client 1 model file has been received, time spent: 303.9753 second
    Client 1 model file size: 204.8664 MB
    Client 1 model file has been decompressed and decrypted, time spent: 2.4281 seconds
    Client 1 model file has been saved to: ./received_models/client_1.joblib
    -------------------------------- Client 1 Completed --------------------------------
    
    Connected to Client 2
    Address: ('127.0.0.1', 52395)
    
    Client 2 model decryption information:
    Key: b'enDtrYvIPnP0S9xZTYJmUUsVdOLT-6FEnoWrNaC4hZA='
    Seed: 0.86141851
    Salt: b'^V{"\x03\xb9\xed\xbcs\xae\xb1\xef\xc9J\x07\x0f'
    
    Client 2 model file has been received, time spent: 280.0054 second
    Client 2 model file size: 205.4604 MB
    Client 2 model file has been decompressed and decrypted, time spent: 2.4964 seconds
    Client 2 model file has been saved to: ./received_models/client_2.joblib
    -------------------------------- Client 2 Completed --------------------------------
    
    Connected to Client 3
    Address: ('127.0.0.1', 52659)
    
    Client 3 model decryption information:
    Key: b'vNcH31fthSJpKVWU80h3DoTM2-yIOjyk7-vvZqnA8ek='
    Seed: 0.86014645
    Salt: b'T\x13G\xc3\x84\xb2SqJs\xca|\xbd_\xba\x90'
    
    Client 3 model file has been received, time spent: 316.0717 second
    Client 3 model file size: 204.4848 MB
    Client 3 model file has been decompressed and decrypted, time spent: 2.7052 seconds
    Client 3 model file has been saved to: ./received_models/client_3.joblib
    -------------------------------- Client 3 Completed --------------------------------
    
    All clients have been processed.
    
    C:\Users>
   ```
2. **Clients running results (second terminal):**
    ```bash
    C:\Users>python client1.py
    Client 1:
    
    Label distribution in the training set: {0: 130615, 1: 21108}
    
    Client 1 model training completed in 30.5484 seconds.
    Client 1 model saved to: ./client_models/client_1.joblib
    
    Client 1 model encryption information:
    Key: b'msXoRJIMUvABzyvm7jtxKImOLwbCNANq8IUrpbMhkWE='
    Seed: 0.86087805
    Salt: b'+\xd6\xdc\xeb\xaa\xf7R\x86\xf5wA\x08\x0f\xcd\xda<'
    
    Model file size: 202.8299 MB
    Model file encrypted, file size: 270.4399 MB, time spent: 1.0805 seconds
    Model file compressed, file size: 204.8664 MB, time spent: 7.8461 seconds
    
    Client 1 has connected to the server.
    Model file has been sent from Client 1, file size: 204.8664 MB, time spent: 0.0234 seconds
    
    C:\Users>python client2.py
    Client 2:
    
    Label distribution in the training set: {0: 130697, 1: 21026}
    
    Client 2 model training completed in 28.8430 seconds.
    Client 2 model saved to: ./client_models/client_2.joblib
    
    Client 2 model encryption information:
    Key: b'enDtrYvIPnP0S9xZTYJmUUsVdOLT-6FEnoWrNaC4hZA='
    Seed: 0.86141851
    Salt: b'^V{"\x03\xb9\xed\xbcs\xae\xb1\xef\xc9J\x07\x0f'
    
    Model file size: 203.4179 MB
    Model file encrypted, file size: 271.2240 MB, time spent: 1.0607 seconds
    Model file compressed, file size: 205.4604 MB, time spent: 7.2820 seconds
    
    Client 2 has connected to the server.
    Model file has been sent from Client 2, file size: 205.4604 MB, time spent: 0.0151 seconds
    
    C:\Users>python client3.py
    Client 3:
    
    Label distribution in the training set: {0: 130504, 1: 21219}
    
    Client 3 model training completed in 38.6214 seconds.
    Client 3 model saved to: ./client_models/client_3.joblib
    
    Client 3 model encryption information:
    Key: b'vNcH31fthSJpKVWU80h3DoTM2-yIOjyk7-vvZqnA8ek='
    Seed: 0.86014645
    Salt: b'T\x13G\xc3\x84\xb2SqJs\xca|\xbd_\xba\x90'
    
    Model file size: 202.4519 MB
    Model file encrypted, file size: 269.9359 MB, time spent: 1.5425 seconds
    Model file compressed, file size: 204.4848 MB, time spent: 10.3495 seconds
    
    Client 3 has connected to the server.
    Model file has been sent from Client 3, file size: 204.4848 MB, time spent: 0.0170 seconds
    
    C:\Users>
    ```



Citations:   
1. [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2. [Real Python: Socket Programming in Python](https://realpython.com/python-sockets/)
3. [Stack Overflow: Send big file over socket](https://stackoverflow.com/questions/56194446/send-big-file-over-socket)
4. [Cryptography: Fernet](https://cryptography.io/en/latest/fernet/)