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
    Address: ('127.0.0.1', 54154)
    
    Client 1 model decryption information:
    Key: b'iXbqbgyvY-nikSVb8E2i9ep5-pUxgwH2NkUbhZULg3k='
    Seed: 0.86039691
    Salt: b'tSk\xc9=`G\x14\xc8\xab\xdb\xe2`\xd3\xd7a'
    
    Client 1 model file has been received, time spent: 227.3206 seconds
    Client 1 model file size: 206.3655 MB
    Client 1 model file has been decompressed and decrypted, time spent: 1.6940 seconds
    Client 1 model file has been saved to: ./received_models/client_1.joblib
    -------------------------------- Client 1 Completed --------------------------------
    
    Connected to Client 2
    Address: ('127.0.0.1', 54175)
    
    Client 2 model decryption information:
    Key: b'QAehMoSTindk6Wzx-Ucy06QunetEQE9pemqJ0mtc_Gs='
    Seed: 0.86010691
    Salt: b'c<YR\x8d\x18\x12\xe5\x8b\x1d\x84(6\xcc\xf9\xbb'
    
    Client 2 model file has been received, time spent: 227.1617 seconds
    Client 2 model file size: 206.6128 MB
    Client 2 model file has been decompressed and decrypted, time spent: 1.6799 seconds
    Client 2 model file has been saved to: ./received_models/client_2.joblib
    -------------------------------- Client 2 Completed --------------------------------
    
    Connected to Client 3
    Address: ('127.0.0.1', 54193)
    
    Client 3 model decryption information:
    Key: b'tyM0V7IVLwvNUv7FFUz15CpaKjD_4j_FdAqbq-Ge6FY='
    Seed: 0.86058145
    Salt: b'\xe5\xac\xac\xe2\\\xf6\x8b\xa8\r\xaa\n\x9a\x1bN~T'
    
    Client 3 model file has been received, time spent: 225.5959 seconds
    Client 3 model file size: 205.4918 MB
    Client 3 model file has been decompressed and decrypted, time spent: 1.6951 seconds
    Client 3 model file has been saved to: ./received_models/client_3.joblib
    -------------------------------- Client 3 Completed --------------------------------
   
    All clients have been processed.
  
    C:\Users>
   ```
2. **Clients running results (second terminal):**
    ```bash
    C:\Users>python client1.py
    Client 1:
    
    Label distribution in the training set: {0: 130542, 1: 21181}
    
    Client 1 model training completed in 28.6346 seconds.
    Client 1 model saved to: ./client_models/client_1.joblib
    
    Client 1 model encryption information:
    Key: b'iXbqbgyvY-nikSVb8E2i9ep5-pUxgwH2NkUbhZULg3k='
    Seed: 0.86039691
    Salt: b'tSk\xc9=`G\x14\xc8\xab\xdb\xe2`\xd3\xd7a'
    
    Model file size: 204.3136 MB
    Model file encrypted, file size: 272.4182 MB, time spent: 1.0880 seconds
    Model file compressed, file size: 206.3655 MB, time spent: 7.3482 seconds
    
    Client 1 has connected to the server.
    Model file has been sent from Client 1, file size: 206.3655 MB, time spent: 0.0156 seconds
    
    C:\Users>python client2.py
    Client 2:
    
    Label distribution in the training set: {0: 130498, 1: 21225}
    
    Client 2 model training completed in 28.8351 seconds.
    Client 2 model saved to: ./client_models/client_2.joblib
    
    Client 2 model encryption information:
    Key: b'QAehMoSTindk6Wzx-Ucy06QunetEQE9pemqJ0mtc_Gs='
    Seed: 0.86010691
    Salt: b'c<YR\x8d\x18\x12\xe5\x8b\x1d\x84(6\xcc\xf9\xbb'
    
    Model file size: 204.5590 MB
    Model file encrypted, file size: 272.7454 MB, time spent: 1.0844 seconds
    Model file compressed, file size: 206.6128 MB, time spent: 7.3530 seconds
    
    Client 2 has connected to the server.
    Model file has been sent from Client 2, file size: 206.6128 MB, time spent: 0.0151 seconds
    
    C:\Users>python client3.py
    Client 3:
    
    Label distribution in the training set: {0: 130570, 1: 21153}
    
    Client 3 model training completed in 28.6886 seconds.
    Client 3 model saved to: ./client_models/client_3.joblib
    
    Client 3 model encryption information:
    Key: b'tyM0V7IVLwvNUv7FFUz15CpaKjD_4j_FdAqbq-Ge6FY='
    Seed: 0.86058145
    Salt: b'\xe5\xac\xac\xe2\\\xf6\x8b\xa8\r\xaa\n\x9a\x1bN~T'
    
    Model file size: 203.4494 MB
    Model file encrypted, file size: 271.2659 MB, time spent: 1.0884 seconds
    Model file compressed, file size: 205.4918 MB, time spent: 7.2975 seconds
    
    Client 3 has connected to the server.
    Model file has been sent from Client 3, file size: 205.4918 MB, time spent: 0.0156 seconds
    
    C:\Users>
    ```



Citations:   
1. [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2. [Real Python: Socket Programming in Python](https://realpython.com/python-sockets/)
3. [Stack Overflow: Send big file over socket](https://stackoverflow.com/questions/56194446/send-big-file-over-socket)
4. [Cryptography: Fernet](https://cryptography.io/en/latest/fernet/)