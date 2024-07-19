from numba import jit,prange
import pandas as pd
import numpy as np
from utils import remove_triangle

# Random initial state
def initial_state(L,string):
    if string == "aligned":
            state = np.full((L,L), 1,dtype=float)
    elif string == "random":
        state = 2 * np.random.randint(2, size=(L,L)) - 1
    else:
        return print("write aligned or random")
    return state

# Total energy calculation
@jit(nopython=True,fastmath=True,nogil=True)
def Total_Energy(config, J):
    L = len(config)
    total_energy = 0
    for i in range(L):
        for j in range(L):
            S = config[i,j]
            nb = config[(i+1)%L, j] + config[i, (j+1)%L] + config[(i-1)%L, j] + config[i, (j-1)%L]
            total_energy += -nb * S
    return (J*total_energy/4) # we need to take of the repeated spins that we counted

# Monte Carlo algorithm
@jit(nopython=True,fastmath=True,nogil=True)
def MC_met(config,beta,J):
    L = len(config)
    a = np.random.randint(0, L)
    b = np.random.randint(0, L)
    sigma =  config[a, b]
    neighbors = config[(a+1)%L, b] + config[a, (b+1)%L] + config[(a-1)%L, b] + config[a, (b-1)%L]
    del_E = 2*sigma*neighbors
    if del_E < 0:
        sigma *= -1
    elif np.random.rand() < np.exp(-del_E*beta):
        sigma *= -1
    config[a, b] = sigma
    return config

# Order Parameter
@jit(nopython=True,fastmath=True,parallel=True)
def mag(config):
    return np.sum(config)

# Create the dynamical model
@jit(nopython=True, fastmath=True, nogil=True)
def temporalseries(T, config, iterations, iterations_fluc, fluctuations, J, n):
    temporal_series = np.zeros((fluctuations, n, n))
    mag_data = np.zeros(fluctuations)
    ene_data = np.zeros(fluctuations)
    beta = 1 / T

    # thermal equilibrium
    for i in range(iterations):
        # if i % 1000000 == 0:
        # print(i/iterations)
        config = MC_met(config, beta, J)

    for z in range(fluctuations):

        for i in range(iterations_fluc):
            config = MC_met(config, beta, J)

        temporal_series[z] = config
        ene_data[z] = Total_Energy(config, J)
        mag_data[z] = mag(config)

    return temporal_series, ene_data, mag_data


# Matrix containing all the system states (Final simulation)
# @jit(nopython=True,fastmath=True,nogil=True)
def Matrix_X(Temps, config, iterations, J, n, block_size, adj_size):
    fluctuations = 200
    corr_size = int((n//block_size)**2)

    X = np.zeros((len(Temps), int((adj_size*adj_size - adj_size)/2)))
    print(X.shape)

    for t in range(len(Temps)):
        print('Models ', t + 1, end="\r", flush=True)

        # Simulating the dynamical model
        model = temporalseries(Temps[t], config, iterations, n * n, fluctuations, J, n)
        # Spatial Average
        avg_model = average_blocks(model, block_size)
        # Transforming into DataFrame
        avg_model_df = pd.DataFrame(avg_model.reshape(fluctuations, avg_model.shape[1] * avg_model.shape[1]))
        # Correlation matrix
        corr_matrix = (pd.DataFrame(avg_model_df).corr())
        # Removing the excess
        corr_matrix = corr_matrix.iloc[int((corr_size - adj_size)/2) : -int((corr_size - adj_size)/2),
                      int((corr_size - adj_size)/2) : - int((corr_size - adj_size)/2)]
        # Removing the lower triangle
        corr_matrix = remove_triangle(corr_matrix)

        # Storing the correlations
        X[t, :] = corr_matrix.reshape(1, int((adj_size*adj_size - adj_size)/2))

    return X

# Spatial Average
def average_blocks(model, block_size):
    time_series_size = len(model[0][:,0,0])
    avg_model = np.zeros((time_series_size,int(len(model[0][1])/block_size),int(len(model[0][1])/block_size)))

    for t in range(time_series_size):
        for i in range(0,(len(model[0][0]) - block_size + 1),block_size):
            for j in range(0, (len(model[0][0]) - block_size + 1),block_size):
                l_0 = block_size
                avg_model[t][i//block_size, j//block_size] = np.mean(model[0][t][i:(i+l_0),j:(j+l_0)])
    return avg_model

# Calculate the correlation spin-spin over the lattice
@jit(nopython=True, fastmath=True, nogil=True)
def corr_net(temporal_series):
    steps = len(temporal_series)
    spins = len(temporal_series[0]) ** 2

    temporal_series_linear = temporal_series.reshape((steps * spins))

    corr_array = np.array([0.0])
    xi = np.zeros(steps)
    xj = np.zeros(steps)
    for i in range(spins):
        corr = np.zeros(spins - (i + 1))
        for j in range(i + 1, spins):
            for n in range(steps):
                xi[n] = temporal_series_linear[i + n * spins]
                xj[n] = temporal_series_linear[j + n * spins]

            diff_i = list()
            diff_j = list()
            for a in range(1, len(xi)):
                value_i = xi[a] - xi[a - 1]
                value_j = xj[a] - xj[a - 1]
                diff_i.append(value_i)
                diff_j.append(value_j)
            corr[j - (i + 1)] = float(np.corrcoef(diff_i, diff_j)[0, 1])

            # corr[j - (i+1)] = float(np.corrcoef(xi,xj)[0,1])
        corr_array = np.concatenate((corr_array, corr))

    return corr_array

J = 1     # J
n = 250    # Lattice size
iterations = ((n*n)*n)    # Iterations to thermal equilibrium

# Temperatures
T_1 = np.linspace(1.7,2.21,750)
T_2 = np.linspace(2.21,2.6,750)
Temps = np.hstack((T_1,T_2)).ravel()


# Simulating the models
config = initial_state(n,"random")

X = Matrix_X(Temps, config,iterations,J,n,13, adj_size=333) # 17, 333

# Save the simulated models
np.savetxt('simulation_corr_matrix_190__250-17.txt', X.ravel())
np.savetxt('temps_190__250-17.txt', Temps.ravel())

