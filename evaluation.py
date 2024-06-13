import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns

# load_pred = np.loadtxt('/Users/rodrigo/Post-Grad/Ising_GNN/Data/y_pred_ising_333_TRUE.txt')
# load_test = np.loadtxt('/Users/rodrigo/Post-Grad/Ising_GNN/Data/y_test_ising_333_TRUE.txt')
#
# load_pred = np.loadtxt('/Users/rodrigo/Post-Grad/Ising_GNN/Data/y_pred_ising_333.txt')
# load_test = np.loadtxt('/Users/rodrigo/Post-Grad/Ising_GNN/Data/y_test_ising_333.txt')

load_pred = np.loadtxt('/Users/rodrigo/Post-Grad/Ising_GNN/Data/y_pred_ising_190_mae.txt')
load_test = np.loadtxt('/Users/rodrigo/Post-Grad/Ising_GNN/Data/y_test_ising_190_mae.txt')

# Fit a regression model
X = sm.add_constant(load_test)  # Adds a constant term to the predictor
model = sm.OLS(load_pred, X)
results = model.fit()

# Predict values along the range of X
x_pred = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
x_pred2 = sm.add_constant(x_pred)
y_pred = results.predict(x_pred2)

# Get confidence intervals
from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_l, iv_u = wls_prediction_std(results, exog=x_pred2, alpha=0.1)  # 95% confidence interval


# plt.figure(figsize=(8, 4), dpi=120)
#
# #Add regression line and confidence interval
# plt.plot(x_pred, y_pred, '-', color='black')  # Reg line
# #plt.fill_between(x_pred, iv_l, iv_u, color='grey', alpha=0.05)  # Confidence interval
#
# # Create the scatter plot
# scatter = plt.scatter(load_test, load_pred, c=load_test, cmap='viridis', alpha=0.5, s=100)
# cbar = plt.colorbar(scatter)
# cbar.set_label('Ising Temperature')
#
# # Plot settings
# plt.title(f'r2 = {r2_score(load_test, load_pred):.2f}')
# plt.ylabel(r'$\hat{T}$')
# plt.xlabel(r'$T$')
# plt.xlim(1.8, 2.4)
# plt.ylim(1.8, 2.4)
# plt.grid()
#
# plt.show()

#####################################################################

# MAE ANALYSIS

# Define segment size
segment_size = 3

# List to store MAE values for each segment
mae_values = []
y_test_seg_mean = []

# Segment the data and calculate MAE for each segment
for start in range(0, len(load_test), segment_size):
    end = start + segment_size
    y_test_segment = load_test[start:end]
    y_pred_segment = load_pred[start:end]
    y_test_seg_mean.append(np.mean(y_test_segment))

    mae_segment = mean_absolute_error(y_test_segment, y_pred_segment)
    mae_values.append(mae_segment)

    print(f"Segment {start // segment_size + 1}:")
    print(f"y_test: {y_test_segment}")
    print(f"y_pred: {y_pred_segment}")
    print(f"MAE: {mae_segment}")
    print()

# Optionally, print all segment-wise MAE values
print("Segment-wise MAE values:", mae_values)

# Optional: Calculate the overall MAE from segment-wise MAEs
overall_mae = np.mean(mae_values)
print("Overall MAE (from segments):", overall_mae)


plt.figure(figsize=(8, 4), dpi=120)

# Create the scatter plot
scatter = plt.scatter(mae_values, y_test_seg_mean, c=y_test_seg_mean, cmap='viridis', alpha=0.5, s=100)
cbar = plt.colorbar(scatter)
cbar.set_label('Ising Temperature')

# Plot settings
plt.title('333 ROIs overall MAE {:.2f}'.format(overall_mae))
plt.ylabel(r'$Real$')
plt.xlabel(r'$MAE$')
plt.xlim(0, 0.3)
#plt.ylim(1.95, 2.4)
plt.grid()

plt.show()