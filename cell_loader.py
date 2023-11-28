import pickle 
from rdkit import Chem
import torch 
from torch.utils.data import Dataset
#from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from models.utils.sparse import from_sparse

def extract_adjacency_matrix_qm9(cc, pad_virtual_nodes=False):
    max_nodes = 9 if pad_virtual_nodes else len(cc.nodes)
    adjacency_matrix = np.zeros((max_nodes, max_nodes))

    for i, j, data in cc.edges(data=True):
        # Retrieve the label of the edge
        label = data.get('label', 0)  # Default to 0 if no label is found
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Since the graph is undirected

    return torch.from_numpy(adjacency_matrix).float()

def extract_node_feature_qm9(cc, pad_virtual_nodes=False):
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

import torch


def extract_edge_features_qm9(cc, pad_virtual_edges=False):
    one_hot_mapping = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    edge_features = []

    for u, v, data in cc.edges(data=True):
        # Retrieve the label of the edge
        label = data.get('label', 1)  # Default to 1 if no label is found
        one_hot = one_hot_mapping.get(label, one_hot_mapping[1])  # Default to label 1 encoding if label is not found
        
        # Add feature for both directions
        edge_features.append(one_hot)  # for (u, v)
        #edge_features.append(one_hot)  # for (v, u)

    if pad_virtual_edges:
        max_edges = 26  # Adjusted for bidirectional edges
        edge_features = edge_features + [[0, 0, 0]] * (max_edges - len(edge_features))

    return torch.tensor(edge_features, dtype=torch.float32)



def extract_up_laplacian_edge(cc, pad_virtual_edges=False):
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
            return torch.zeros((cc.number_of_edges(), cc.number_of_edges()))
        return torch.from_numpy(cc.up_laplacian_matrix(rank=1, signed = False).todense()).float()

def extract_low_laplacian_edge(cc, pad_virtual_edges=False):
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

def to_edge_index(edge_dict):
    # Extract edge pairs (keys of the dictionary)
    edge_pairs = list(edge_dict.keys())

    # Separate source and destination nodes
    source_nodes = [edge[0] for edge in edge_pairs]
    dest_nodes = [edge[1] for edge in edge_pairs]

    # Convert to PyTorch tensor
    edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)

    return edge_index


# create DataLoader for cell complex by padding and masking
class CCDataset(Dataset):
    def __init__(self, cell_complexes):
        self.cell_complexes = cell_complexes
        # self.max_edges = max([len(cell_complex.edges) for cell_complex in cell_complexes])
        # self.max_nodes = max([len(cell_complex.nodes) for cell_complex in cell_complexes])
        
    def __len__(self):
        return len(self.cell_complexes)
    
    # get x_0, x_1, a_0, upper_a_1, lower_a_1
    def __getitem__(self, idx):
        cc = self.cell_complexes[idx]
        x_0 = extract_node_feature_qm9(cc)
        x_1 = extract_edge_features_qm9(cc, pad_virtual_edges=False)
        a_0 = extract_adjacency_matrix_qm9(cc)
        upper_l_1 = extract_up_laplacian_edge(cc, pad_virtual_edges=False)
        lower_l_1 = extract_low_laplacian_edge(cc, pad_virtual_edges=False)
        edge_index = to_edge_index(cc.get_edge_attributes('label'))
        y = cc.name['gap']
        
        return x_0, x_1, a_0, upper_l_1, lower_l_1,y, edge_index

def cc_to_data(cc):
    x_0 = extract_node_feature_qm9(cc)
    x_1 = extract_edge_features_qm9(cc, pad_virtual_edges=False)
    a_0 = extract_adjacency_matrix_qm9(cc)
    upper_l_1 = extract_up_laplacian_edge(cc, pad_virtual_edges=False)
    lower_l_1 = extract_low_laplacian_edge(cc, pad_virtual_edges=False)
    #edge_index = to_edge_index(cc.get_edge_attributes('label'))
    y = cc.name['gap']
    
    return x_0, x_1, a_0, upper_l_1, lower_l_1,y




# version that can utilize topomodelx functions 
def cc_to_data_topomodelx(cc):
    x_0 = extract_node_feature_qm9(cc, pad_virtual_nodes=False)
    x_1 = extract_edge_features_qm9(cc, pad_virtual_edges=False)
    a_0 = cc.adjacency_matrix(rank=0)
    a_0 = torch.from_numpy(a_0.todense())
    lower_l_1 = cc.down_laplacian_matrix(rank=1, signed=False)
    lower_l_1 = torch.from_numpy(lower_l_1.todense())
    try:
        upper_l_1 = cc.up_laplacian_matrix(rank=1, signed=False)
        upper_l_1 = torch.from_numpy(upper_l_1.todense())
    except:
        upper_l_1 = np.zeros(
            (lower_l_1.shape[0], lower_l_1.shape[0])
        )
        upper_l_1 = torch.from_numpy(upper_l_1)
    y = cc.name['gap']
    
    return x_0, x_1, a_0, upper_l_1, lower_l_1,y


class CellData(Data): 
    def __init__(self, x_0, x_1, a_0, upper_l_1, lower_l_1, y):
        super(CellData, self).__init__()
        self.x_0 = x_0
        self.x_1 = x_1
        self.a_0 = a_0
        self.upper_l_1 = upper_l_1
        self.lower_l_1 = lower_l_1
        self.y = y
        #self.edge_index = edge_index
 
def custom_collate(batch):

    max_nodes = sum([data.a_0.shape[0] for data in batch])
    max_edges = sum([data.lower_l_1.shape[0] for data in batch]) 
    
    current_node_index = 0
    current_edge_index = 0
    node_indices = []
    edge_indices = []

    # Initialize large matrices
    large_a_0 = torch.zeros((max_nodes, max_nodes))
    large_upper_l_1 = torch.zeros((max_edges, max_edges))
    large_lower_l_1 = torch.zeros((max_edges, max_edges))
    
    # Initialize lists for other features and indices
    batch_x_0 = []
    batch_x_1 = []
    batch_y = []

    for i, data in enumerate(batch):
        x_0 = data.x_0
        x_1 = data.x_1
        a_0 = data.a_0
        lower_l_1 = data.lower_l_1
        upper_l_1 = data.upper_l_1
        y = data.y

        # Number of nodes and edges in the current graph
        num_nodes = a_0.shape[0]
        num_edges = lower_l_1.shape[0]

        # Place the matrices in the large matrices
        large_a_0[current_node_index:current_node_index+num_nodes, current_node_index:current_node_index+num_nodes] = a_0
        large_upper_l_1[current_edge_index:current_edge_index+num_edges, current_edge_index:current_edge_index+num_edges] = upper_l_1
        large_lower_l_1[current_edge_index:current_edge_index+num_edges, current_edge_index:current_edge_index+num_edges] = lower_l_1

        # convert adjacency matrix to sparse_coo_tensor
        
        # Update indices
        node_indices.extend([i] * num_nodes)
        edge_indices.extend([i] * num_edges)

        # Append other features
        batch_x_0.append(x_0)
        batch_x_1.append(x_1)
        batch_y.append(y)

    # Convert lists to tensors or appropriate format
    batch_x_0 = torch.cat(batch_x_0, dim=0)
    batch_x_1 = torch.cat(batch_x_1, dim=0)
    batch_y = torch.tensor(batch_y, dtype=torch.float)
    
    large_a_0 = large_a_0.to_sparse()
    large_upper_l_1 = large_upper_l_1.to_sparse()
    large_lower_l_1 = large_lower_l_1.to_sparse()

    return batch_x_0, large_a_0, batch_x_1, large_lower_l_1, large_upper_l_1, batch_y, node_indices, edge_indices


# with open('data/qm9_test_cell_complex.pkl', 'rb') as f:
#     qm9_test_cell_complex = pickle.load(f)



# ccs = [CellData(*cc_to_data_topomodelx(cc)) for cc in qm9_test_cell_complex]
# # samples = ccs[0:4]
# # print(custom_collate(samples))


# print("Creating DataLoader...")
# loader = DataLoader(ccs, batch_size=3, shuffle=False, collate_fn=custom_collate)
# print("DataLoader created.")





# # # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch in loader:
#     print(batch)
#     break
        