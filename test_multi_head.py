from models.can.can_layer import MultiHeadCellAttention_v2
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pickle
from cell_loader import CCDataset
from models.can.can import CAN
from models.utils.sparse import from_sparse

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("data/qm9_test_cell_complex.pkl", "rb") as f:
    cc_list = pickle.load(f)
cc_list = cc_list[0:40]

from cell_loader import CCDataset
dataset = CCDataset(cc_list)


x_0_list = [data[0] for data in dataset]
x_1_list = [data[1] for data in dataset]
y_list = [random.choice([0, 1]) for _ in range(30)]

lower_neighborhood_list = []
upper_neighborhood_list = []
adjacency_0_list = []

for cell_complex in cc_list:
    adjacency_0 = cell_complex.adjacency_matrix(rank=0)
    adjacency_0 = torch.from_numpy(adjacency_0.todense()).to_sparse()
    adjacency_0_list.append(adjacency_0)

    lower_neighborhood_t = cell_complex.down_laplacian_matrix(rank=1)
    lower_neighborhood_t = from_sparse(lower_neighborhood_t)
    lower_neighborhood_list.append(lower_neighborhood_t)

    try:
        upper_neighborhood_t = cell_complex.up_laplacian_matrix(rank=1)
        upper_neighborhood_t = from_sparse(upper_neighborhood_t)
    except:
        upper_neighborhood_t = np.zeros(
            (lower_neighborhood_t.shape[0], lower_neighborhood_t.shape[0])
        )
        upper_neighborhood_t = torch.from_numpy(upper_neighborhood_t).to_sparse()

    upper_neighborhood_list.append(upper_neighborhood_t)


from cell_loader import CCDataset
dataset = CCDataset(cc_list)

mh = MultiHeadCellAttention_v2(in_channels=3, out_channels = 32, 
                               heads = 3, concat= True,
                               att_activation= torch.nn.ReLU(), aggr_func='sum',
                               dropout=0.5)
i = 17

print('number of edges:',x_1_list[i].shape[0])
(target_index_i,source_index_j,) = upper_neighborhood_list[i].indices() 
print(target_index_i.shape)
print('max index:',target_index_i.max().item())
print(mh(x_1_list[i], upper_neighborhood_list[i]).shape)