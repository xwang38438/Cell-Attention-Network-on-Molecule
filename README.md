# Cell-Attention-Network-on-Molecule
Test if higher-order data structure can perform better than general GNNs on molecular graphs

# Installation 

1. follow the instruction in _TopoModelX_
2. install graph-tool by `conda install graph-tool=2.58=py311h35b2f40_1 -c conda-forge`

# Data Preprocessing 

To preprocess the molecular graph datasets for training models, run the following command:

`python data/preprocess.py --dataset QM9`
`python data/preprocess_for_nspdk.py --dataset QM9`

and then run the cells in ``


