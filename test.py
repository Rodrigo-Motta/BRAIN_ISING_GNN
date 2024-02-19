import pandas as pd
import utils as ut
import numpy as np
from model import GCN
import torch

# FOR HEAD MOTION
merged_df,merged_phenotypic  = ut.read_motion()

# Health subjects
merged_phenotypic = merged_phenotypic[merged_phenotypic['DX'] != 0]

#merged_df = merged_df.set_index(['Institution', 'Subject', 'Run'])
#merged_phenotypic = merged_phenotypic.set_index(['Institution', 'Subject', 'Run'])
common_indices = merged_df.index.intersection(merged_phenotypic.index)
merged_phenotypic = merged_phenotypic.loc[common_indices,:]
merged_df = merged_df.loc[common_indices,:]
merged_df = merged_df.reset_index()
merged_phenotypic = merged_phenotypic.reset_index()


X_fmri = merged_df.iloc[:,14:]
y = merged_df['Max Motion (mm)']
y = merged_phenotypic['Age']


A = ut.reconstruct_symmetric_matrix(190, X_fmri.iloc[:,:].mean(axis=0))
train_data, val_data = ut.create_graph(X_fmri, X_fmri, y, y,method={'knn_group' : ut.compute_KNN_graph(A, 15)})#, method={'threshold': 0.8})
train_loader, val_loader = ut.create_batch(train_data, val_data, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = GCN(190, 3).to(device)
model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/model.pth'))
model.eval()

y_pred_aux_age = []
for y_i in val_loader:
    y_pred_aux_age.append((model(y_i))[1].detach().numpy().ravel()[0])

my_dict_age = {}
for i in range(len(y.values)):
    my_dict_age[y.values[i]] = y_pred_aux_age[i]
