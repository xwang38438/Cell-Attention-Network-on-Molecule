import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from models.can.can import CAN
from models.utils.sparse import from_sparse
torch.manual_seed(0)
np.random.seed(0)
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("data/qm9_test_cell_complex.pkl", "rb") as f:
    cc_list = pickle.load(f)
    
from cell_loader import CCDataset
dataset = CCDataset(cc_list)

x_0_list = [data[0] for data in dataset]
x_1_list = [data[1] for data in dataset]
y_list = [data[5] for data in dataset]

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

in_channels_0 = x_0_list[0].shape[-1]
in_channels_1 = x_1_list[0].shape[-1]

model = CAN(
    in_channels_0,
    in_channels_1,
    32,
    dropout=0.5,
    heads=2,
    num_classes=1,
    n_layers=2,
    att_lift=True,
)
model = model.to(device)


crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)


test_size = 0.3
x_1_train, x_1_test = train_test_split(x_1_list, test_size=test_size, shuffle=False)
x_0_train, x_0_test = train_test_split(x_0_list, test_size=test_size, shuffle=False)
lower_neighborhood_train, lower_neighborhood_test = train_test_split(
    lower_neighborhood_list, test_size=test_size, shuffle=False
)
upper_neighborhood_train, upper_neighborhood_test = train_test_split(
    upper_neighborhood_list, test_size=test_size, shuffle=False
)
adjacency_0_train, adjacency_0_test = train_test_split(
    adjacency_0_list, test_size=test_size, shuffle=False
)
y_train, y_test = train_test_split(y_list, test_size=test_size, shuffle=False)



test_interval = 1
num_epochs = 50
for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    num_samples = 0
    correct = 0
    model.train()
    for x_0, x_1, adjacency, lower_neighborhood, upper_neighborhood, y in zip(
        x_0_train,
        x_1_train,
        adjacency_0_train,
        lower_neighborhood_train,
        upper_neighborhood_train,
        y_train,
    ):
        x_0 = x_0.float().to(device)
        x_1, y = x_1.float().to(device), torch.tensor(y, dtype=torch.long).to(device)
        adjacency = adjacency.float().to(device)
        lower_neighborhood, upper_neighborhood = lower_neighborhood.float().to(
            device
        ), upper_neighborhood.float().to(device)
        opt.zero_grad()
        y_hat = model(x_0, x_1, adjacency, lower_neighborhood, upper_neighborhood)
        loss = crit(y_hat, y)
        correct += (y_hat.argmax() == y).sum().item()
        num_samples += 1
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())
    train_acc = correct / num_samples
    print(
        f"Epoch: {epoch_i} loss: {np.mean(epoch_loss):.4f} Train_acc: {train_acc:.4f}",
        flush=True,
    )
    if epoch_i % test_interval == 0:
        with torch.no_grad():
            num_samples = 0
            correct = 0
            for x_0, x_1, adjacency, lower_neighborhood, upper_neighborhood, y in zip(
                x_0_test,
                x_1_test,
                adjacency_0_test,
                lower_neighborhood_test,
                upper_neighborhood_test,
                y_test,
            ):
                x_0 = x_0.float().to(device)
                x_1, y = x_1.float().to(device), torch.tensor(y, dtype=torch.long).to(
                    device
                )
                adjacency = adjacency.float().to(device)
                lower_neighborhood, upper_neighborhood = lower_neighborhood.float().to(
                    device
                ), upper_neighborhood.float().to(device)
                y_hat = model(
                    x_0, x_1, adjacency, lower_neighborhood, upper_neighborhood
                )
                
                correct += (y_hat.argmax() == y).sum().item()
                num_samples += 1
            test_acc = correct / num_samples
            print(f"Test_acc: {test_acc:.4f}", flush=True)