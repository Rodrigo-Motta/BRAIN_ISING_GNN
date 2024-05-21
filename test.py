import pandas as pd
import utils as ut
import numpy as np
from model import GCN
#from model_190 import GCN

import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
def reconstruct_symmetric_matrix(size, upper_triangle_array, diag=1):
    '''
    Reconstruct symmetric matrix from the upper triangle array
    '''
    # Create an empty matrix filled with zeros
    result = np.zeros((size, size))

    # Fill the upper triangle of the matrix (including the diagonal) with the values from the input array
    result[np.triu_indices_from(result)] = upper_triangle_array

    # Make the matrix symmetric by adding its transpose
    result = result + result.T

    # Fill the diagonal with the specified diagonal value (default is 1)
    np.fill_diagonal(result, diag)

    return result

# ADHD 333
df = pd.read_csv('/Users/rodrigo/Post-Grad/adhd_dataset_gordon.csv')
N = 333

# Health subjects
df = df[df['DX'] == 0]
df = df[df[' Max Rotation (degree)'] < 1]
df = df[df['Max Motion (mm)'] < 1]

X_fmri = df.iloc[:,1:-37]

y = df['Age']





# FOR HEAD MOTION
# merged_df,merged_phenotypic  = ut.read_motion()
# N = 190
#
# # Health subjects
# merged_phenotypic = merged_phenotypic[merged_phenotypic['DX'] == 0]
# merged_phenotypic = merged_phenotypic[merged_phenotypic[' Max Rotation (degree)'] < 5]
# merged_phenotypic = merged_phenotypic[merged_phenotypic['Max Motion (mm)'] < 5]
#
#
# #merged_df = merged_df.set_index(['Institution', 'Subject', 'Run'])
# #merged_phenotypic = merged_phenotypic.set_index(['Institution', 'Subject', 'Run'])
# common_indices = merged_df.index.intersection(merged_phenotypic.index)
# merged_phenotypic = merged_phenotypic.loc[common_indices,:]
# merged_df = merged_df.loc[common_indices,:]
# merged_df = merged_df.reset_index()
# merged_phenotypic = merged_phenotypic.reset_index()
#
#
# X_fmri = merged_df.iloc[:,14:]
#
# # Shuffle
# #X_fmri = X_fmri.sample(frac=1, axis=1)  # `frac=1` means to return all columns
#
# y = merged_df['Max Motion (mm)']
# y = merged_phenotypic['Age']


A = ut.reconstruct_symmetric_matrix(N, X_fmri.iloc[:,:].mean(axis=0))
train_data, val_data = ut.create_graph(X_fmri, X_fmri, y, y, size=N, method={'knn_group' : ut.compute_KNN_graph(A, 15)})#, method={'threshold': 0.8})
train_loader, val_loader = ut.create_batch(train_data, val_data, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = GCN(N, 3).to(device)
#model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/Data/model_params_190.pth'))
#model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/Data/model_params_190_test.pth'))
#model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/Data/model_params_190_mae.pth'))
model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/Data/model_params_333_TRUE.pth'))

model.eval()

y_pred_aux_age = []
for y_i in val_loader:
    y_pred_aux_age.append((model(y_i))[1].detach().numpy().ravel()[0])

# my_dict_age = {}
# for i in range(len(y.values)):
#     my_dict_age[y.values[i]] = y_pred_aux_age[i]

y = pd.DataFrame(y)
y['y_pred'] = y_pred_aux_age
y = y.dropna()

plt.figure(dpi=120)
sns.regplot(data=y, x='Age', y='y_pred',color='black')

correlation_value = np.corrcoef([y['Age'].values, y['y_pred'].values])[0,1]
spearman = stats.spearmanr(y['Age'].values, y['y_pred'].values).statistic

text = f'Pearson: {correlation_value:.2f}$^{{****}}$ \nSpearman: {spearman:.2f}$^{{****}}$'

# Adding text box with correlation value
plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

#plt.ylim(2.1,2.5)
plt.ylabel('Estimated Ising Temperature', size=16)
plt.xlabel(r'Age', size=16)
plt.tight_layout()
plt.show()
#
# plt.figure(figsize=(10,4),dpi=120)
#
# plt.scatter(pd.DataFrame(y['Age']),pd.DataFrame(y['y_pred']),
#             c=merged_df[' Max Rotation (degree)'], cmap='viridis_r', alpha=0.5, s=20)
#
# y = y.dropna()
# correlation_value = np.corrcoef([y['Age'].values, y['y_pred'].values])[0,1]
# spearman = stats.spearmanr(y['Age'].values, y['y_pred'].values).statistic
#
# text = f'Pearson: {correlation_value:.2f}$^{{****}}$ \nSpearman: {spearman:.2f}$^{{****}}$'
#
# # # Adding text box with correlation value
# plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
#          bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
#
# plt.colorbar(label=' Max Rotation (degree)')
# plt.xlabel(r'Age')
# plt.ylabel(r'$Estimated Ising Temperature$')
# #plt.xlim(1.8,2.5)
# #plt.ylim(1.8,2.5)
# plt.grid()
#
# plt.show()