import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils as ut
from scipy import stats

sns.set_palette("Set2")

# ADHD 333
df = pd.read_csv('/Users/rodrigo/Post-Grad/adhd_dataset_gordon.csv')
N = 333

# Health subjects
#df = df[df['DX'] == 0]
df = df[df[' Max Rotation (degree)'] < 3]
df = df[df['Max Motion (mm)'] < 3]
df = df.rename(columns={'Gender' : 'Sex'})


# FOR HEAD MOTION
#merged_df,merged_phenotypic  = ut.read_motion()

# Health subjects
#merged_phenotypic = merged_phenotypic[merged_phenotypic['DX'] == 0]

# common_indices = merged_df.index.intersection(merged_phenotypic.index)
# merged_phenotypic = merged_phenotypic.loc[common_indices,:]
# merged_df = merged_df.loc[common_indices,:]
# merged_df = merged_df.reset_index()
# merged_phenotypic = merged_phenotypic.reset_index().rename(columns={'Gender' : 'Sex'})

# sns.violinplot(df.replace({0.0 : 'Male', 1.0 : 'Female'}), y='Age', x='Sex')
# plt.xlabel('Sex', size=14)
# plt.ylabel('Age', size=14)
# ---------------------------------------------------------------
# sns.violinplot(df
#                .replace({0 : 'Typically Developing Children', 1 : 'ADHD-Combined', 2 : 'ADHD-Hyperactive/Impulsive', 3 : 'ADHD-Inattentive'}),
#                y='Age', x='DX')
# plt.xlabel('DX', size=14)
# plt.ylabel('Age', size=14)
# ------------------------------------------------------------------
# sns.lmplot(df,x='Age', y='Max Motion (mm)', line_kws={'color': 'black'}, scatter_kws={'color':'black'})
# plt.ylabel('Max Motion (mm)', size=14)
# plt.xlabel('Age', size=14)
# correlation_value = stats.pearsonr(df.Age, df['Max Motion (mm)']).statistic
# p_value = stats.pearsonr(df.Age, df['Max Motion (mm)']).pvalue
# text = f'Pearson: {correlation_value:.2f}$^{{}}$ \n $p$: {p_value:.2f}$^{{}}$'
#
# plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
#          bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'), size=14)
# ----------------------------------------------------------------
# sns.lmplot(df,x='Age', y=' Max Rotation (degree)', line_kws={'color': 'black'}, scatter_kws={'color':'black'})
# plt.ylabel(' Max Rotation(degree)', size=14)
# plt.xlabel('Age', size=14)
# correlation_value = stats.pearsonr(df.Age, df[' Max Rotation (degree)']).statistic
# p_value = stats.pearsonr(df.Age, df[' Max Rotation (degree)']).pvalue
# text = f'Pearson: {correlation_value:.2f}$^{{}}$ \n $p$: {p_value:.2f}$^{{}}$'
#
# plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
#          bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'), size=14)
#
# plt.show()
