import utils as ut
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from model.model import GCN
import torch

# ROIs
N = 333 

# Number of simulations
num_simulations = 1500 

######### Change to your data path #########

# Simulation correlation matrix data path
simulation_path = '/Users/rodrigo/Documents/Projects/BRAIN_ISING_GNN/simulation_corr_matrix_revision.txt'  
# Simulation Temperature data path
temp_path = '/Users/rodrigo/Documents/Projects/BRAIN_ISING_GNN/temps_revision.txt'

loss = nn.L1Loss() #nn.MSELoss() #nn.L1Loss()


def TRAIN_LOSS(loader, loss):
    model.eval()
    l1_weight = 0

    pred = []
    label = []

    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)[1]
        l1_penalty = l1_weight * sum([p.abs().sum() for p in model.parameters()])
        loss_value = loss(output.squeeze(), (data.y).float())
        loss_with_penalty = loss_value + l1_penalty
        loss_all += data.num_graphs * loss_with_penalty.item()
        pred.append(output)
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    mae = mean_absolute_error(y_pred, y_true)
    mse = mean_squared_error(y_pred, y_true)

    return mae, mse, loss_all / len(train_data)


def GCN_train(loader, loop, loss):
    model.train()
    l1_weight = 0
    loss_all = 0
    pred = []
    label = []

    for (x, y) in enumerate(loop):
        y = y.to(device)
        optimizer.zero_grad()
        output = model(y)[1]
        # Adding L1 regularization
        l1_penalty = l1_weight * sum([p.abs().sum() for p in model.parameters()])
        loss_value = loss(output.squeeze(), (y.y).float())
        loss_with_penalty = loss_value + l1_penalty
        loss_with_penalty.backward()
        optimizer.step()
        scheduler.step()
        loss_all += y.num_graphs * loss_with_penalty.item()
        pred.append(output)
        label.append(y.y)

        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss_all / len(train_data))

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    mae = mean_absolute_error(y_pred, y_true)
    mse = mean_squared_error(y_pred, y_true)

    return mae, mse, loss_all / len(train_data)


def GCN_test(loader, loss):
    model.eval()
    l1_weight = 0

    pred = []
    label = []

    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)[1]
        l1_penalty = l1_weight * sum([p.abs().sum() for p in model.parameters()])
        loss_value = loss(output.squeeze(), (data.y).float())
        loss_with_penalty = loss_value + l1_penalty
        loss_all += data.num_graphs * loss_with_penalty.item()
        pred.append(output)
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    mae = mean_absolute_error(y_pred, y_true)
    mse = mean_squared_error(y_pred, y_true)

    return mae, mse, loss_all / len(val_data)

print('############ Start Training ############')

X = np.loadtxt(simulation_path).ravel().reshape(num_simulations, int((N*N - N)/2))
Temps = np.loadtxt(temp_path)

# Filter low temperatures
X = X[150:]
Temps = Temps[150:]

df = pd.concat([pd.DataFrame(X),pd.DataFrame(Temps)],axis=1).sample(frac=1)

X_t = np.array(df.iloc[:150, :-1].values, dtype=np.float32)
Temps_t = np.array(df.iloc[:150, -1].values, dtype=np.float32)

X = np.array(df.iloc[150:, :-1].values, dtype=np.float32)
Temps = np.array(df.iloc[150:, -1].values, dtype=np.float32)

X = np.array(X, dtype=np.float32)
Temps = np.array(Temps, dtype=np.float32)

# Split Test Set
#X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X_t), pd.DataFrame(Temps_t),test_size=0.95, shuffle=True, random_state=42)
train_data, test_data = ut.create_graph(pd.DataFrame(X_t), pd.DataFrame(X_t), pd.DataFrame(Temps_t), pd.DataFrame(Temps_t), size=N, method={'knn' : 15})
train_loader, test_loader = ut.create_batch(train_data, test_data, batch_size=64)

# Split train and validation set
X_train, X_val, y_train, y_val = train_test_split(pd.DataFrame(X),pd.DataFrame(Temps),test_size=0.15, shuffle=True, random_state=42)

train_data, val_data = ut.create_graph(X_train, X_val, y_train, y_val,size=N, method={'knn' : 15})

train_loader, val_loader = ut.create_batch(train_data, val_data, batch_size=64)

metrics = {"loss_train": [], "loss_test": [], "mae_test": [], "mae_train": []}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(N, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)  # ,momentum=0.35)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=5e-2, cycle_momentum=False)

for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

# model.apply(init_weights)

min_v_loss = np.inf

NUM_EPOCHS = 100

for epoch in range(1, NUM_EPOCHS + 1):
    loop = tqdm(train_loader)
    train_mae, train_mse, train_loss = GCN_train(train_loader, loop, loss)
    test_mae, test_mse, test_loss = GCN_test(val_loader, loss)
    TRAIN_mae, TRAIN_mse, TRAIN_loss = TRAIN_LOSS(train_loader, loss)

    # scheduler.step()

    metrics['loss_train'].append(TRAIN_loss)
    metrics['loss_test'].append(test_loss)
    metrics['mae_test'].append(test_mae)
    metrics['mae_train'].append(TRAIN_mae)

    print('Val MAE {} , Val Loss {}'.format(test_mae, test_loss))
    print('Train MAE {} , Train Loss {}'.format(TRAIN_mae, TRAIN_loss))

    if test_mae < .07:
        break


# Save the model parameters to a file
torch.save(model.state_dict(), '../model_params_revision_mae.pth')

# Create a DataFrame from the dictionary
df = pd.DataFrame(metrics)

# Save the DataFrame to a CSV file
df.to_csv('../model_eval_revision_mae.csv', index=False)

# Save predictions
y_pred_aux = np.array([])
y_test_aux = []
for y_i in test_loader:
    y_pred_aux = np.concatenate((y_pred_aux, (model(y_i))[1].detach().numpy().ravel()) )
    y_test_aux.append(y_i.y.numpy()[0])

np.savetxt('../y_pred_ising_revision_mae.txt', y_pred_aux.ravel())
np.savetxt('../y_test_ising_revision_mae.txt', np.array(y_test_aux).ravel())
