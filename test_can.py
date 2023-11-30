import random

import numpy as np
import torch
#from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from cell_loader import CCDataset, cc_to_data_topomodelx, CellData, custom_collate
from torch.utils.data import DataLoader, random_split
from models.can.can import CAN
from models.utils.sparse import from_sparse
import pickle
torch.manual_seed(0)
np.random.seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# record time for loading data
import time
start_time = time.time()

# ------------------- Load data ------------------- #
with open('data/qm9_train_cell_complex.pkl', 'rb') as f:
    qm9_train_cell_complex = pickle.load(f)

with open('data/qm9_test_cell_complex.pkl', 'rb') as f:
    qm9_test_cell_complex = pickle.load(f)

train_ccs = [CellData(*cc_to_data_topomodelx(cc)) for cc in qm9_train_cell_complex]
test_ccs = [CellData(*cc_to_data_topomodelx(cc)) for cc in qm9_test_cell_complex]

# split train_ccs into train and val
train_ccs, val_ccs = random_split(train_ccs, [int(0.8*len(train_ccs)), len(train_ccs)-int(0.8*len(train_ccs))])

print('code running till here')

train_loader = DataLoader(train_ccs, batch_size=1024, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_ccs, batch_size=128, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_ccs, batch_size=128, shuffle=False, collate_fn=custom_collate)



print("Time for Loading Data: %s" % (time.time() - start_time))

# ------------------- Load model ------------------- #
model = CAN(
    in_channels_0 = 4,
    in_channels_1 = 3,
    out_channels= 32,
    dropout=0.5,
    heads=3,
    n_layers=3,
    att_lift=True,
).to(device)

# ------create optimizer and loss function ------ #
crit = torch.nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------- Training ------------------- #
test_interval = 5
num_epochs = 1
for epoch_i in range(1, num_epochs+1):
    epoch_loss = []
    
    print(f'Epoch {epoch_i}')
    model.train()
    for batch in train_loader:
        batch_x_0, batch_a_0, batch_x_1, batch_lower_l_1, batch_upper_l_1, batch_y, edge_indices = batch

        batch_x_0 = batch_x_0.float().to(device)
        #print('batch_x_0.shape',batch_x_0.shape)
        batch_a_0 = batch_a_0.float().to(device)
        #print('batch_a_0.shape',batch_a_0.shape)
        batch_x_1 = batch_x_1.float().to(device)
        #print('batch_x_1.shape',batch_x_1.shape)
        batch_lower_l_1 = batch_lower_l_1.float().to(device)
        batch_upper_l_1 = batch_upper_l_1.float().to(device)
        batch_y = batch_y.unsqueeze(1).to(device)
        #print('edge_indices',edge_indices)
        #print('batch_a_0 \n', batch_a_0.to_dense())
        #print('lower_l_1 \n', batch_lower_l_1.to_dense())
        #source, target = batch_a_0.indices()
        #print(len(source))

        # ------------------- Forward pass ------------------- #
        opt.zero_grad()
        
        y_hat = model(
            batch_x_0, batch_x_1, batch_a_0, batch_lower_l_1, batch_upper_l_1, edge_indices
        )
        
        loss = crit(y_hat, batch_y)
        print('loss',loss.item())
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())
    
    print(
        f"Epoch: {epoch_i} loss: {np.mean(epoch_loss):.4f}",
        flush=True,
    )
    # ------------------- Validation ------------------- #
    model.eval()
    val_loss = []
    for batch in val_loader:
        batch_x_0, batch_a_0, batch_x_1, batch_lower_l_1, batch_upper_l_1, batch_y, edge_indices = batch

        batch_x_0 = batch_x_0.float().to(device)
        batch_a_0 = batch_a_0.float().to(device)
        batch_x_1 = batch_x_1.float().to(device)
        batch_lower_l_1 = batch_lower_l_1.float().to(device)
        batch_upper_l_1 = batch_upper_l_1.float().to(device)
        batch_y = batch_y.unsqueeze(1).to(device)

        with torch.no_grad():
            y_hat = model(
                batch_x_0, batch_x_1, batch_a_0, batch_lower_l_1, batch_upper_l_1, edge_indices
            )
            loss = crit(y_hat, batch_y)
            val_loss.append(loss.item())
    print(
        f"Epoch: {epoch_i} val loss: {np.mean(val_loss):.4f}",
        flush=True,
    )
    
    
    if epoch_i % test_interval == 0:
        model.eval()
        test_loss = []
        for batch in test_loader:
            batch_x_0, batch_a_0, batch_x_1, batch_lower_l_1, batch_upper_l_1, batch_y, edge_indices = batch

            batch_x_0 = batch_x_0.float().to(device)
            batch_a_0 = batch_a_0.float().to(device)
            batch_x_1 = batch_x_1.float().to(device)
            batch_lower_l_1 = batch_lower_l_1.float().to(device)
            batch_upper_l_1 = batch_upper_l_1.float().to(device)
            batch_y = batch_y.unsqueeze(1).to(device)

            with torch.no_grad():
                y_hat = model(
                    batch_x_0, batch_x_1, batch_a_0, batch_lower_l_1, batch_upper_l_1, edge_indices
                )
                loss = crit(y_hat, batch_y)
                test_loss.append(loss.item())
        print(
            f"Epoch: {epoch_i} test loss: {np.mean(test_loss):.4f}",
            flush=True,
        )
