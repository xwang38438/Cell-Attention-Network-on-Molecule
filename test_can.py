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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:',device)

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

# --- hyperparameters --- #
out_channels_list = [16]
heads_list = [2]#[2, 3, 4]
n_layers_list = [2] #[1, 2, 3]

result_dict = {}

# ------------------- Load model ------------------- #

# record training time
start_time = time.time()

for out_channels in out_channels_list:
    for heads in heads_list:
        for n_layers in n_layers_list:
            print(f'out_channels: {out_channels}, heads: {heads}, n_layers: {n_layers}')
            model = CAN(
                in_channels_0 = 4,
                in_channels_1 = 3,
                out_channels= out_channels,
                dropout=0.0,
                heads=heads,
                n_layers=n_layers,
                att_lift=True,
            )
            model.to(device)

            # ------create optimizer and loss function ------ #
            crit = torch.nn.L1Loss()
            opt = torch.optim.Adam(model.parameters(), lr=0.003)

            # ------------------- Training ------------------- #

            best_val_loss = float('inf')
            best_model_path = f'check_points/best_can_model_hid{out_channels}_head{heads}_{n_layers}layers.pth'

            train_losses = []
            val_losses = []
            test_losses = []


            test_interval = 5
            num_epochs = 70
            for epoch_i in range(1, num_epochs+1):
                epoch_loss = []
                
                print(f'Epoch {epoch_i}')
                model.train()
                for batch in train_loader:
                    batch_x_0, batch_a_0, batch_x_1, batch_lower_l_1, batch_upper_l_1, batch_y, edge_indices = batch

                    batch_x_0 = batch_x_0.float().to(device)
                    batch_a_0 = batch_a_0.float().to(device)
                    batch_x_1 = batch_x_1.float().to(device)
                    batch_lower_l_1 = batch_lower_l_1.float().to(device)
                    batch_upper_l_1 = batch_upper_l_1.float().to(device)
                    batch_y = batch_y.unsqueeze(1).to(device)

                    # ------------------- Forward pass ------------------- #
                    opt.zero_grad()
                    
                    y_hat = model(
                        batch_x_0, batch_x_1, batch_a_0, batch_lower_l_1, batch_upper_l_1, edge_indices
                    )
                    
                    loss = crit(y_hat, batch_y)
                    loss.backward()
                    opt.step()
                    epoch_loss.append(loss.item())
                
                train_losses.append(np.mean(epoch_loss))
                print(
                    f"Epoch: {epoch_i} Train loss: {np.mean(epoch_loss):.4f}",
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
                        
                val_losses.append(np.mean(val_loss))
                print(
                    f"Epoch: {epoch_i} val loss: {np.mean(val_loss):.4f}",
                    flush=True,
                )
                current_val_loss = np.mean(val_loss)
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(model.state_dict(), best_model_path)
                
                
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
                    
                    test_losses.append(np.mean(test_loss))
                    print(
                        f"Epoch: {epoch_i} test loss: {np.mean(test_loss):.4f}",
                        flush=True,
                    )
            # time end
            
            losses_dict = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_losses': test_losses,
                'best_val_loss': best_val_loss,
                'time': time.time() - start_time,
            }
            
            result_dict[f'out_channels: {out_channels}, heads: {heads}, n_layers: {n_layers}'] = losses_dict


# Save to a file
with open('check_points/can_losses_dict.pkl', 'wb') as f:
    pickle.dump(result_dict, f)

print("Losses dictionary saved successfully.")