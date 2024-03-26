import pandas as pd
import utils as ut
import numpy as np
from model import GCN
import torch

phenotypic = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/phenotypic.csv')
phenotypic = phenotypic.set_index('subject')

df_1 = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/corr_matrices.csv')
df_1 = df_1.set_index('subject')

outliers = np.loadtxt('/Users/rodrigo/Post-Grad/BHRC/subjects_outliers.txt').astype(int)
df_1 = df_1.drop(df_1.loc[outliers,:].index)

df_1 = df_1.drop(df_1.loc[[10019, 10081, 10095, 10132, 10143, 10242, 10758, 20158, 20494, 20608],:].index) # WHY ?
phenotypic = phenotypic.loc[df_1.index, :]

df_1.iloc[:,:-1] = (df_1.iloc[:,:-1])
df_1 = df_1[df_1.iloc[:,1:-1].max(axis=1) > .3]

df_1['age_mri_baseline'] = phenotypic['age_mri_baseline']/12#/phenotypic['TOTAL_DAWBA'].max()

df_1 = df_1.reset_index()
X_fmri_1 = pd.DataFrame(df_1.drop(columns=['age_mri_baseline','subject']).values)
y_1 =  pd.DataFrame((df_1['age_mri_baseline'].drop(columns=['age_mri_baseline','subject'])))

# ----------------------------------------------------------------------------------------

phenotypic = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/phenotypic.csv')
phenotypic = phenotypic.set_index('subject')

df_2 = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/corr_matrices_wave2.csv')
df_2 = df_2.set_index('subject')

outliers = np.loadtxt('/Users/rodrigo/Post-Grad/BHRC/subjects_outliers_wave2.txt').astype(int)
df_2 = df_2.drop(df_2.loc[outliers,:].index)


df_2 = df_2.drop(df_2.loc[[10068, 20415],:].index)
phenotypic = phenotypic.loc[df_2.index, :]

df_2.iloc[:,:-1] = (df_2.iloc[:,:-1])
df_2 = df_2[df_2.iloc[:,1:-1].max(axis=1) > .3]

df_2['age_mri_wave2'] = phenotypic['age_mri_wave2']/12#/phenotypic['TOTAL_DAWBA'].max()

df_2 = df_2.reset_index()
X_fmri_2 = pd.DataFrame(df_2.drop(columns=['age_mri_wave2','subject']).values)
y_2 =  pd.DataFrame((df_2['age_mri_wave2'].drop(columns=['age_mri_wave2','subject'])))

X_fmri = pd.concat([X_fmri_1, X_fmri_2])
y = pd.concat([y_1, y_2])
#

A = ut.reconstruct_symmetric_matrix(333, X_fmri.iloc[:,:].mean(axis=0))
train_data, val_data = ut.create_graph(X_fmri, X_fmri, y, y,method={'knn_group' : ut.compute_KNN_graph(A, 15)})#, method={'threshold': 0.8})
train_loader, val_loader = ut.create_batch(train_data, val_data, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = GCN(333, 3).to(device)
model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/model.pth'))
model.eval()

y_pred_aux_age = []
for y_i in val_loader:
    y_pred_aux_age.append((model(y_i))[1].detach().numpy().ravel()[0])

my_dict_age = {}
for i in range(len(y.values)):
    my_dict_age[y.values[i]] = y_pred_aux_age[i]
