# MCS Project
pending update...   

### Getting Started
1. This project contains following source code files:
    ```
    helper.py
    server.py
    client1.py
    client2.py
    client3.py
    run.bat
    ```


### Running

1. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Running the Code:**

    You have two options to run the code:

   - **Option 1: Using `run.bat`:**

     Double-click on `run.bat` file. This will automatically execute the necessary scripts.

   - **Option 2: Using Command Line:**

     If you prefer to run the scripts manually, execute the following commands in your terminal:

     1. Loading the helper `helper.py`:
        ```bash
        python helper.py
        ```
     2. Start the server `server.py`:
        ```bash
        python server.py
        ```
     3. Run three clients in three terminals:
        ```bash
        python client1.py
        ```
        ```bash
        python client2.py
        ```
        ```bash
        python client3.py
        ```




### Evaluating

1. **Example of running results on the server:**
   ```bash
    Server:

    Server is waiting for connections...
    
    Connected to Client 1
    Address: ('127.0.0.1', 50889)
    
    Connected to Client 2
    Address: ('127.0.0.1', 50890)
    
    Connected to Client 3
    Address: ('127.0.0.1', 50891)
    
    Model file has been received from Client 1 and saved on ./models/model_client_1.joblib.
    Time taken to receive model file from Client 1: 0.817179 seconds.
    
    Model file has been received from Client 2 and saved on ./models/model_client_2.joblib.
    Time taken to receive model file from Client 2: 0.766375 seconds.
    
    Model file has been received from Client 3 and saved on ./models/model_client_3.joblib.
    Time taken to receive model file from Client 3: 0.723068 seconds.
   ```
2. **Example of running results on the client:**
    ```bash
    Client 1:
    
    Label distribution in the training set: {0: 133998, 1: 22059}
    Label distribution in the testing set: {0: 33556, 1: 5459}
    
    Client 1 model training completed in 38.708024 seconds.
    ---------------------
    Accuracy : 0.93451237
    Precision: 0.93368981
    Recall   : 0.93451237
    F1 Score : 0.92811258
    ---------------------
    Client 1 model saved locally.
    Client 1 has connected to the server.
    Client ID has been sent from Client 1.
    Model file size has been sent from Client 1.
    Model file has been sent from Client 1.
    Time taken to send model file: 0.091080 seconds.
    ```



Author: Hongwei Zhang

Citations:   
1. [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2. [Real Python: Socket Programming in Python](https://realpython.com/python-sockets/)
3. [Stack Overflow: Send big file over socket](https://stackoverflow.com/questions/56194446/send-big-file-over-socket)