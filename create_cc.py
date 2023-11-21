import pickle
import argparse
from utils.cell_lifting import graph_to_edge_index, get_rings
from toponetx.classes.cell_complex import CellComplex
import time
import numpy as np
# create a pkl file with cell complex data
# cell complex stores atom, bond, and ring information
# each cell complex is associated with a molecule with its energy


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='QM9', choices=['ZINC250k', 'QM9'])
args = parser.parse_args()
dataset = args.dataset

time_start = time.time()

if dataset == 'QM9':
    with open(f'data/{dataset.lower()}_train_nx.pkl', 'rb') as f:
        qm9_train = pickle.load(f)
    with open(f'data/{dataset.lower()}_test_nx.pkl', 'rb') as f:
        test_mols = pickle.load(f)
    
 
    # extract ring from each molecular graph 
    mol_train_rings = []
    mol_test_rings = []
    
    qm9_train_cell_complex = []
    qm9_test_cell_complex = []
    
    for graph in qm9_train:
        attribute = {'gap': graph.graph['gap'], 'U0': graph.graph['U0']}
        edge_index = graph_to_edge_index(graph)
        if edge_index.nelement() == 0:
            mol_test_rings.append([])
            continue
        rings = get_rings(edge_index)
        # turn each molecular graph into a cell complex 
        cell_complex = CellComplex(graph, attribute)
        cell_complex.add_cells_from(rings, rank = 2)
        
        mol_train_rings.append(rings)
        qm9_train_cell_complex.append(cell_complex)
        
    for graph in test_mols:
        attribute = {'gap': graph.graph['gap'], 'U0': graph.graph['U0']}
        edge_index = graph_to_edge_index(graph)
        if edge_index.nelement() == 0:
            mol_test_rings.append([])
            continue
        rings = get_rings(edge_index)
        cell_complex = CellComplex(graph, attribute)
        cell_complex.add_cells_from(rings, rank = 2)
        
        mol_test_rings.append(rings)
        qm9_test_cell_complex.append(cell_complex)
        
    with open(f'data/{dataset.lower()}_train_cell_complex.pkl', 'wb') as f:
        pickle.dump(qm9_train_cell_complex, f)
    with open(f'data/{dataset.lower()}_test_cell_complex.pkl', 'wb') as f:
        pickle.dump(qm9_test_cell_complex, f)


    

