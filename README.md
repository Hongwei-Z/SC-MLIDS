## SC-MLIDS: Fusion-based Machine Learning Framework for Intrusion Detection in Wireless Sensor Networks


### Abstract:
This paper proposes the Server–Client Machine Learning Intrusion Detection System (SC-MLIDS), a novel fusion framework designed to enhance security in Wireless Sensor Networks (WSNs), which are inherently vulnerable to various security threats due to their distributed nature and resource constraints. Traditional Intrusion Detection Systems (IDSs) often face challenges with high computational demands and privacy issues. SC-MLIDS addresses these problems by integrating Federated Learning (FL) with a multi-sensor fusion approach to implementing two layers of defence that operate independently of specific attack types. Moreover, this framework leverages a server–client architecture to efficiently manage and process data from sensor nodes, sink nodes, and gateways within the network. The core innovation of SC-MLIDS lies in its dual model aggregation algorithms at the gateway: one assesses model performance and weight, while the other uses majority voting to integrate predictions from both client and server models. As a result, this approach reduces redundant data transmissions and enhances detection accuracy, making it more effective than conventional methods in WSNs. Our proposed framework outperforms current state-of-the-art techniques, achieving F1-scores of 99.78% and 98.80% for the two aggregation algorithms, namely, Weighted Score and Majority Voting. This validation demonstrates the effectiveness of SC-MLIDS in providing accurate intrusion detection and robust data management.

### File Structure:

1. **Core Modules:**
  - [`helper.py`](helper.py): Dataset processing, model metrics calculation, and encryption key generation.
  - [`server.py`](server.py): Server implementation for receiving, decrypting, decompressing client data and training network traffic models.
  - [`client.py`](client.py): Client implementation for local sensor model training, data encryption, compression, and transmission.
  - [`algorithm.py`](algorithm.py): Implementation of aggregate prediction algorithms (Weighted Score and Majority Voting).

2. **Demonstrations:**
  - [`Demo/Aggregate-1.ipynb`](Demo/Aggregate-1.ipynb): Showcase of the two aggregate prediction algorithms.
  - [`Demo/Aggregate-2.ipynb`](Demo/Aggregate-2.ipynb): Extended demonstration using multiple distinct classifiers.

3. **System Logs:**
  - [`Log/Server.txt`](Log/Server.txt), [`Log/Client.txt`](Log/Client.txt): Server and client terminal logs with sample execution results.

4. **Dataset Resources:**
  - [`Dataset/Preprocessing.ipynb`](Dataset/Preprocessing.ipynb): Data preprocessing workflow.
  - [`Dataset/merged_data.csv`](Dataset/merged_data.csv): Processed dataset generated from preprocessing.


### Code Execution Instruction:
1. Install the required packages:
   `pip install -r requirements.txt`
1. Start the server and clients, train the models, transfer the models, and save the models. Open two terminals, one for the server and another for the clients.   
     1. In the first terminal, load the helper and start the server:   
         ```bash   
         python helper.py
         python server.py
         ```   
      
     2. In the second terminal, run client:   
         ```bash
         python client.py
         ```
 
2. Aggregate Prediction Algorithms:
   ```text
   See 'Demo/Aggregate-1.ipynb' and 'Demo/Aggregate-2.ipynb'.
   ```


### How to Cite

If you use this project, please cite our paper:

IEEE Citation:      
```text
H. Zhang, D. Upadhyay, M. Zaman, A. Jain, and S. Sampalli, “SC-MLIDS: Fusion-based Machine Learning Framework for Intrusion Detection in Wireless Sensor Networks,” Ad Hoc Networks, vol. 175, p. 103871, 2025, doi:10.1016/j.adhoc.2025.103871.
```

BibTeX: 
```text
@article{ZHANG2025103871,
  title   = {SC-MLIDS: Fusion-based Machine Learning Framework for Intrusion Detection in Wireless Sensor Networks},
  author={Zhang, Hongwei and Upadhyay, Darshana and Zaman, Marzia and Jain, Achin and Sampalli, Srinivas},
  journal = {Ad Hoc Networks},
  volume  = {175},
  pages   = {103871},
  year    = {2025},
  issn    = {1570-8705},
  doi     = {10.1016/j.adhoc.2025.103871},
  url     = {https://www.sciencedirect.com/science/article/pii/S1570870525001192},
}
```


Author: Hongwei Zhang

Updated on 2025-05-04

EOF

