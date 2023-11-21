from time import time
import pickle
import json
import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from utils.mol_utils import mols_to_nx, smiles_to_mols


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='QM9', choices=['ZINC250k', 'QM9'])
args = parser.parse_args()

dataset = args.dataset
start_time = time()

with open(f'data/valid_idx_{dataset.lower()}.json') as f:
    test_idx = json.load(f)

if dataset == 'QM9':
    test_idx = test_idx['valid_idxs']
    test_idx = [int(i) for i in test_idx]
    col = 'SMILES1'
    
    smiles = pd.read_csv(f'data/{dataset.lower()}.csv')[col]
    gap_df = pd.read_csv(f'data/{dataset.lower()}.csv')['gap']
    U0_df = pd.read_csv(f'data/{dataset.lower()}.csv')['U0']

    test_smiles = [smiles.iloc[i] for i in test_idx]
    test_gap = [gap_df.iloc[i] for i in test_idx]
    test_U0 = [U0_df.iloc[i] for i in test_idx]
    train_smiles = [smiles.iloc[i] for i in range(len(smiles)) if i not in test_idx]
    train_gap = [gap_df.iloc[i] for i in range(len(gap_df)) if i not in test_idx]
    train_U0 = [U0_df.iloc[i] for i in range(len(U0_df)) if i not in test_idx]

    nx_graphs_test = mols_to_nx(smiles_to_mols(test_smiles))
    nx_graphs_train = mols_to_nx(smiles_to_mols(train_smiles))

    # assign graph-level target values
    for i, g in enumerate(nx_graphs_test):
        g.graph['gap'] = test_gap[i]
        g.graph['U0'] = test_U0[i]
        
    for i, g in enumerate(nx_graphs_train):
        g.graph['gap'] = train_gap[i]
        g.graph['U0'] = train_U0[i]


    print(f'Converted the test molecules into {len(nx_graphs_test)} graphs')
    print(f'Converted the train molecules into {len(nx_graphs_train)} graphs')

    with open(f'data/{dataset.lower()}_test_nx.pkl', 'wb') as f:
        pickle.dump(nx_graphs_test, f)

    with open(f'data/{dataset.lower()}_train_nx.pkl', 'wb') as f:
        pickle.dump(nx_graphs_train, f)

    print(f'Total {time() - start_time:.2f} sec elapsed')
    
elif dataset == 'ZINC250k':
    col = 'smiles'
    col_pred = 'logP'
else:
    raise ValueError(f"[ERROR] Unexpected value data_name={dataset}")

