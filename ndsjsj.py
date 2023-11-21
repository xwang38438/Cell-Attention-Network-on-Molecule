import pickle 
from rdkit import Chem
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np

def extract_adjacency_matrix_qm9(cc, pad_virtual_nodes=True):
    max_nodes = 9 if pad_virtual_nodes else len(cc.nodes)
    adjacency_matrix = np.zeros((max_nodes, max_nodes))

    for i, j, data in cc.edges(data=True):
        # Retrieve the label of the edge
        label = data.get('label', 0)  # Default to 0 if no label is found
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Since the graph is undirected

    return torch.from_numpy(adjacency_matrix).float()

def extract_node_feature_qm9(cc, pad_virtual_nodes=True):
    one_hot_mapping = {'C': [1, 0, 0, 0], 'N': [0, 1, 0, 0], 'O': [0, 0, 1, 0], 'F': [0, 0, 0, 1]}
    if pad_virtual_nodes:
        max_nodes = 9
    else:
        max_nodes = len(cc.nodes)
    # Initialize a matrix of zeros
    feature_matrix = np.zeros((max_nodes, 4))

    for node, data in cc.nodes(data=True):
        # Retrieve the label and convert to one-hot encoding
        label = data.get('label', 'C')  # Default to 'C' if no label is found
        one_hot = one_hot_mapping.get(label, one_hot_mapping['C'])  # Default to 'C' encoding if label is not found
        feature_matrix[node] = one_hot

    return torch.from_numpy(feature_matrix).float()

def extract_edge_features_qm9(cc, pad_virtual_edges=True):
    
    one_hot_mapping = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    edge_features = []

    for _,_, data in cc.edges(data=True):
        # Retrieve the label of the edge
        label = data.get('label', 1)  # Default to 1 if no label is found
        one_hot = one_hot_mapping.get(label, one_hot_mapping[1])  # Default to label 1 encoding if label is not found
        edge_features.append(one_hot)
        
    if pad_virtual_edges:
        max_edges = 13
        edge_features = edge_features + [[0,0,0]] * (max_edges - len(edge_features))

    return torch.tensor(edge_features, dtype=torch.float32)

def extract_up_laplacian_edge(cc, pad_virtual_edges=True):
    if pad_virtual_edges:
        max_edges = 13
        if cc.number_of_cells() == 0:
            return torch.zeros((max_edges, max_edges))
        not_pad_upl1 = cc.up_laplacian_matrix(rank=1, signed = False).todense()
        not_pad_upl1 = np.array(not_pad_upl1)
        pad_upl1 = np.zeros((max_edges, max_edges))
        pad_upl1[:not_pad_upl1.shape[0], :not_pad_upl1.shape[1]] = not_pad_upl1
        
        return torch.from_numpy(pad_upl1).float()
    else:
        if cc.number_of_cells() == 0:
            return torch.zeros((0,0))
        return torch.from_numpy(cc.up_laplacian_matrix(rank=1, signed = False).todense()).float()

def extract_low_laplacian_edge(cc, pad_virtual_edges=True):
    if pad_virtual_edges:
        max_edges = 13
        if cc.number_of_edges() == 0:
            raise ValueError('No edges in the cell complex')
        not_pad_lowl1 = cc.down_laplacian_matrix(rank=1, signed = False).todense()
        not_pad_lowl1 = np.array(not_pad_lowl1)
        pad_lowl1 = np.zeros((max_edges, max_edges))
        pad_lowl1[:not_pad_lowl1.shape[0], :not_pad_lowl1.shape[1]] = not_pad_lowl1
        
        return torch.from_numpy(pad_lowl1).float()
    else: 
        if cc.number_of_edges() == 0:
            raise ValueError('No edges in the cell complex')
        return torch.from_numpy(cc.down_laplacian_matrix(rank=1, signed = False).todense()).float()


with open('data/qm9_test_cell_complex.pkl', 'rb') as f:
    qm9_test_cell_complex = pickle.load(f)

cc = qm9_test_cell_complex[0]
    
print(cc.get_node_attributes('label'))
print(cc.get_edge_attributes('label'))
#print(extract_edge_features_qm9(qm9_train_cell_complex[0]))
#print(cc.up_laplacian_matrix(rank=1, signed = False).todense())
print(extract_low_laplacian_edge(cc))



# create DataLoader for cell complex by padding and masking
class CCDataset(Dataset):
    def __init__(self, cell_complexes):
        self.cell_complexes = cell_complexes
        self.max_edges = max([len(cell_complex.edges) for cell_complex in cell_complexes])
        self.max_nodes = max([len(cell_complex.nodes) for cell_complex in cell_complexes])
        
    def __len__(self):
        return len(self.cell_complexes)
    
    # get x_0, x_1, a_0, upper_a_1, lower_a_1
    def __getitem__(self, idx):
        cc = self.cell_complexes[idx]
        x_0 = extract_node_feature_qm9(cc)
        x_1 = extract_edge_features_qm9(cc)
        a_0 = extract_adjacency_matrix_qm9(cc)
        upper_l_1 = extract_up_laplacian_edge(cc)
        lower_l_1 = extract_low_laplacian_edge(cc)
        
        return x_0, x_1, a_0, upper_l_1, lower_l_1
            
            
dataset = CCDataset(qm9_test_cell_complex)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(batch)
    break
        