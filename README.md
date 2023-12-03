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
    Server is waiting for connections ...
    ------------------------------------------------------------------------------------
    Connected to Client 1
    Address: ('127.0.0.1', 54514)
    
    Client 1 model decryption information:
    Key: b'vIF5q7--Qrr2d3dJn0rWkWW0ZBnipkQy4gSSGmVbEwE='
    Seed: 0.09660938
    Salt: b'\x18\xfai\xbd\xb8\xf9\xb7M\xcd4\x0f\xe0<Y\x1f\xfc'
    
    Client 1 model file has been received, time spent: 289.2070 seconds
    Client 1 model file size: 231.7189 MB
    Client 1 model file has been decompressed and decrypted, time spent: 1.9532 seconds
    Client 1 model file has been saved to: ./received_models/client_1.joblib
    -------------------------------- Client 1 Completed --------------------------------
    
    Connected to Client 2
    Address: ('127.0.0.1', 54589)
    
    Client 2 model decryption information:
    Key: b'sl69JS77LAkzRJVrRciLA-VT3TLFmGEvv-Qud8hZhaU='
    Seed: 0.09761581
    Salt: b'\x1a-\xf1\x11\xbd\x88C\xe1L,_\x16H\x8e\xb8\xa2'
    
    Client 2 model file has been received, time spent: 290.6902 seconds
    Client 2 model file size: 232.3311 MB
    Client 2 model file has been decompressed and decrypted, time spent: 1.9053 seconds
    Client 2 model file has been saved to: ./received_models/client_2.joblib
    -------------------------------- Client 2 Completed --------------------------------
    
    Connected to Client 3
    Address: ('127.0.0.1', 54602)
    
    Client 3 model decryption information:
    Key: b'1YwESbc6Bui9W0Bls45OeUPMhaBJQNOIm5v1rF_1Egg='
    Seed: 0.09700314
    Salt: b'\xdd\xe0>\x01\xd2\xc6\x92\x88\x11d\xc6\xc1<\xe3g~'
    
    Client 3 model file has been received, time spent: 289.5289 seconds
    Client 3 model file size: 231.6734 MB
    Client 3 model file has been decompressed and decrypted, time spent: 1.9433 seconds
    Client 3 model file has been saved to: ./received_models/client_3.joblib
    -------------------------------- Client 3 Completed --------------------------------
    
    -------------------------- All Clients Have Been Processed -------------------------
    Training the global model using network traffic data ...
    Label distribution in the training set: {0: 211854, 1: 1967129}
    
    Global model training completed in 217.9969 seconds.
    Global model saved to: ./received_models/global_model.joblib
    ------------------------------------- All Done -------------------------------------
    
    C:\Users>
   ```
2. **Clients running results (second terminal):**
    ```bash
    C:\Users>python client1.py
    Client 1:
    
    Label distribution in the training set: {0: 70170, 1: 656157}
    
    Client 1 model training completed in 70.6325 seconds.
    Client 1 model saved to: ./client_models/client_1.joblib
    
    Client 1 model encryption information:
    Key: b'vIF5q7--Qrr2d3dJn0rWkWW0ZBnipkQy4gSSGmVbEwE='
    Seed: 0.09660938
    Salt: b'\x18\xfai\xbd\xb8\xf9\xb7M\xcd4\x0f\xe0<Y\x1f\xfc'
    
    Model file size: 229.4157 MB
    Model file encrypted, file size: 305.8877 MB, time spent: 1.2273 seconds
    Model file compressed, file size: 231.7189 MB, time spent: 8.2409 seconds
    
    Client 1 has connected to the server.
    Model file has been sent from Client 1, file size: 231.7189 MB, time spent: 0.0160 seconds
    
    C:\Users>python client2.py
    Client 2:
    
    Label distribution in the training set: {0: 70901, 1: 655426}
    
    Client 2 model training completed in 71.3592 seconds.
    Client 2 model saved to: ./client_models/client_2.joblib
    
    Client 2 model encryption information:
    Key: b'sl69JS77LAkzRJVrRciLA-VT3TLFmGEvv-Qud8hZhaU='
    Seed: 0.09761581
    Salt: b'\x1a-\xf1\x11\xbd\x88C\xe1L,_\x16H\x8e\xb8\xa2'
    
    Model file size: 230.0215 MB
    Model file encrypted, file size: 306.6954 MB, time spent: 1.2224 seconds
    Model file compressed, file size: 232.3311 MB, time spent: 8.3474 seconds
    
    Client 2 has connected to the server.
    Model file has been sent from Client 2, file size: 232.3311 MB, time spent: 0.0159 seconds
    
    C:\Users>python client3.py
    Client 3:
    
    Label distribution in the training set: {0: 70456, 1: 655871}
    
    Client 3 model training completed in 70.8952 seconds.
    Client 3 model saved to: ./client_models/client_3.joblib
    
    Client 3 model encryption information:
    Key: b'1YwESbc6Bui9W0Bls45OeUPMhaBJQNOIm5v1rF_1Egg='
    Seed: 0.09700314
    Salt: b'\xdd\xe0>\x01\xd2\xc6\x92\x88\x11d\xc6\xc1<\xe3g~'
    
    Model file size: 229.3701 MB
    Model file encrypted, file size: 305.8269 MB, time spent: 1.2158 seconds
    Model file compressed, file size: 231.6734 MB, time spent: 8.3175 seconds
    
    Client 3 has connected to the server.
    Model file has been sent from Client 3, file size: 231.6734 MB, time spent: 0.0162 seconds
    
    C:\Users>
    ```



Citations:   
1. [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2. [Real Python: Socket Programming in Python](https://realpython.com/python-sockets/)
3. [Stack Overflow: Send big file over socket](https://stackoverflow.com/questions/56194446/send-big-file-over-socket)
4. [Cryptography: Fernet](https://cryptography.io/en/latest/fernet/)