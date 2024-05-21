import pandas as pd
import utils as ut
import numpy as np
from model import GCN
import torch
import matplotlib.pyplot as plt

phenotypic = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/phenotypic.csv')
phenotypic = phenotypic.set_index('subject')

df_1 = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/corr_matrices.csv')
df_1 = df_1.set_index('subject')

subjects = np.loadtxt('/Users/rodrigo/Post-Grad/BHRC/DadosRodrigo/CARAS12.txt').astype(int)
df_1 = df_1.loc[subjects,:]

phenotypic = phenotypic.loc[df_1.index, :]

phenotypic = phenotypic[phenotypic.TOTAL_DAWBA == 0.0]

df_1 = df_1[df_1.index.isin(phenotypic.index)]

#df_1['age_mri_baseline'] = phenotypic['age_mri_baseline']/12#/phenotypic['TOTAL_DAWBA'].max()
df_1['TOTAL_DAWBA'] = phenotypic['TOTAL_DAWBA']/12

df_1 = df_1.reset_index()
X_fmri_1 = pd.DataFrame(df_1.drop(columns=['TOTAL_DAWBA','subject']).values) #age_mri_wave
y_1 =  pd.DataFrame((df_1['TOTAL_DAWBA'].drop(columns=['TOTAL_DAWBA','subject']))) #age_mri_wave

# ---------------------------------------------------------------------------------------
phenotypic = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/phenotypic.csv')
phenotypic = phenotypic.set_index('subject')

df_2= pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/corr_matrices.csv')
df_2 = df_2.set_index('subject')

subjects = np.loadtxt('/Users/rodrigo/Post-Grad/BHRC/DadosRodrigo/CARAS12.txt').astype(int)
df_2 = df_2.loc[subjects,:]

phenotypic = phenotypic.loc[df_2.index, :]

phenotypic = phenotypic[phenotypic.TOTAL_DAWBA != 0]

df_2 = df_2[df_2.index.isin(phenotypic.index)]

df_2['TOTAL_DAWBA'] = phenotypic['TOTAL_DAWBA']#/12#/phenotypic['TOTAL_DAWBA'].max()

df_2 = df_2.reset_index()
X_fmri_2 = pd.DataFrame(df_2.drop(columns=['TOTAL_DAWBA','subject']).values)
y_2 =  pd.DataFrame((df_2['TOTAL_DAWBA'].drop(columns=['TOTAL_DAWBA','subject'])))


# ----------------------------------------------------------------------------------------

# phenotypic = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/phenotypic.csv')
# phenotypic = phenotypic.set_index('subject')
#
# df_2 = pd.read_csv('/Users/rodrigo/Post-Grad/BHRC/corr_matrices_wave2.csv')
# df_2 = df_2.set_index('subject')
#
# subjects = np.loadtxt('/Users/rodrigo/Post-Grad/BHRC/DadosRodrigo/CARAS12.txt').astype(int)
# df_2 = df_2.loc[subjects,:]
#
# phenotypic = phenotypic.loc[df_2.index, :]
#
# phenotypic = phenotypic[phenotypic.TOTAL_DAWBA == 0.0]
#
# df_2 = df_2[df_2.index.isin(phenotypic.index)]
#
# df_2['age_mri_wave2'] = phenotypic['age_mri_wave2']/12#/phenotypic['TOTAL_DAWBA'].max()
#
# df_2 = df_2.reset_index()
# X_fmri_2 = pd.DataFrame(df_2.drop(columns=['age_mri_wave2','subject']).values)
# y_2 =  pd.DataFrame((df_2['age_mri_wave2'].drop(columns=['age_mri_wave2','subject'])))

############################################################################

X_fmri = pd.concat([X_fmri_1, X_fmri_2])
y = pd.DataFrame(np.concatenate([y_1.values, y_2.values]).ravel())

A = ut.reconstruct_symmetric_matrix(333, X_fmri.iloc[:,:].mean(axis=0))
train_data, val_data = ut.create_graph(X_fmri, X_fmri, y, y, size=333, method={'knn_group' : ut.compute_KNN_graph(A, 15)})#, method={'threshold': 0.8})
train_loader, val_loader = ut.create_batch(train_data, val_data, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = GCN(333, 3).to(device)
model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/Data/model_params_333_TRUE.pth'))
model.eval()

y_pred_aux_age = []
for y_i in val_loader:
    y_pred_aux_age.append((model(y_i))[1].detach().numpy().ravel()[0])

my_dict_age = {}
for i in range(len(y.values)):
    my_dict_age[y.values[i][0]] = y_pred_aux_age[i]


y = pd.DataFrame(y)
y['y_pred'] = y_pred_aux_age
y = y.dropna()

# y_pred_aux_age_1 = np.array(y_pred_aux_age[:160])
# y_pred_aux_age_2 = np.array(y_pred_aux_age[160:]) #309
#
# y_1 = np.array(y[:160])
# y_2 = np.array(y[160:])