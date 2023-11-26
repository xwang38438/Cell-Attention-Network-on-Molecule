from models.can.can_layer import *
import torch 
import pickle
from cell_loader import cc_to_data, CellData
# set seed 
torch.manual_seed(0)

with open('data/qm9_test_cell_complex.pkl', 'rb') as f:
    qm9_test_cell_complex = pickle.load(f)

cc = qm9_test_cell_complex[0]

cc_data = CellData(*cc_to_data(cc))
x_0 = cc_data.x_0
x_1 = cc_data.x_1
a_0 = cc_data.a_0
# turn a_0 to sparse coodinate tensor
# a_0 = a_0.nonzero(as_tuple=False).t().contiguous()
a_0 = a_0.to_sparse()
# print(a_0.indices())
# print(x_1)
# print(a_0)
up1 = cc_data.upper_l_1
low1 = cc_data.lower_l_1

print(up1.shape, low1.shape)


# print(x_0.shape, x_1.shape, a_0.shape)

# liftlayer = LiftLayer(in_channels_0=4, heads = 3, 
#                       signal_lift_activation=torch.nn.ReLU,
#                       signal_lift_dropout= 0.0)

# liftlayer.forward(x_0, a_0)