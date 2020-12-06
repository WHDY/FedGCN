# FedGCN

### File Description

- ***dataWithGCNSetting***: Cora， Citeseer and PubMed. The dataset is split as training, validate, and test following the prior work (Kipf GCN).

- ***dtaGenerator.py***: to read dataset
- ***Model.py***: GCN model
- ***centeralizedTrain.py***:  train GCN in centeralized scenario.
- ***utils.py***:  includes some functions, such as the function to process adjacency matrix.
- ***clients***:  clients end
- ***sever.py***:  the server end to train GCN in Federated scenario.
- ***experimentsOfFed.py & experimentsOfLEpochs***: experiments codes

