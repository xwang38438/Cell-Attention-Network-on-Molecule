from models.can.can_layer import CANLayer, MultiHeadLiftLayer, PoolLayer
import torch 
import pickle
from cell_loader import cc_to_data
# set seed 
torch.manual_seed(0)


with open('data/qm9_test_cell_complex.pkl', 'rb') as f:
    qm9_test_cell_complex = pickle.load(f)

# cc = qm9_test_cell_complex[0]
# cc_data = cc_to_data(cc)
# x_0 = cc_data.x_0
# x_1 = cc_data.x_1
# a_0 = cc_data.a_0

# print(x_0)
# print(x_1)
# print(a_0)

# multihead_lift_layer = MultiHeadLiftLayer(in_channels_0=4, heads=1)

# lifted_features = multihead_lift_layer(x_0, neighborhood_0_to_0, x_1)
# lifted_features.shape, lifted_features