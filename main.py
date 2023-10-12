import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx

from cmrc import CMRC
from data import ks_pseudospectral
from mrc import MRC
from rc import ReservoirComputer
from lyapunov_exponents import compute_lyapunov_exponents
from ks_lyapunov_exponents import compute_lyapunov_exponents

"""
Generate Data
"""
print("generating data")
# Parameters
L = 44
N = 100
T = 200
dt = 0.025
t = np.linspace(0, T, int(T/dt)) 

# Initial condition: some random perturbation around a sine wave
x = np.linspace(0, L, N, endpoint=False)
u0 = np.exp(-((x - L/2)**2)/10)

u_hist = odeint(ks_pseudospectral, u0, t, args=(L, N))

"""
Partition Data
"""
print("partitioning data")
train_size = int(0.8 * len(t))
train_data = u_hist[:train_size]
test_data = u_hist[train_size:]

"""
Instantiate Models
"""
print("instantiating models")
dim_system = N
dim_reservoir = 1500
rho = 0.9     
sigma = 0.5      
density = 0.05 

rc = ReservoirComputer(dim_system, dim_reservoir, rho, sigma, density)
mrc = MRC(dim_system, dim_reservoir, rho, sigma, density)
cmrc = CMRC(dim_system, dim_reservoir, rho, sigma, density)

"""
Train Models
"""
print("train models")
rc.train(train_data)
mrc.train(train_data)
cmrc.train(train_data)

"""
Forecast
"""
print("forecasting")
forecast_steps = len(test_data)
predictions_rc = rc.predict(forecast_steps)
predictions_mrc = mrc.predict(forecast_steps)
predictions_cmrc = cmrc.predict(forecast_steps)

"""
Visualize Predictions
"""
print("generating visualizations")

fig, ax = plt.subplots(3, 3, figsize=(15, 10))

# Truth plots (1st column)
for i in range(3):
    ax[i][0].imshow(test_data.T, aspect='auto', origin='lower', cmap='jet', extent=[0, T, 0, L])
    ax[i][0].set_title('Truth')
    ax[i][0].set_ylabel('x')
    ax[i][0].set_xlabel('Time')
    fig.colorbar(ax[i][0].get_images()[0], ax=ax[i][0], location='right')

# Predictions plots (2nd column)
models_predictions = [predictions_rc.T, predictions_mrc.T, predictions_cmrc.T]
model_names = ['RC', 'MRC', 'CMRC']
for i, predictions in enumerate(models_predictions):
    ax[i][1].imshow(predictions, aspect='auto', origin='lower', cmap='jet', extent=[0, T, 0, L])
    ax[i][1].set_title(model_names[i] + ' Prediction')
    ax[i][1].set_ylabel('x')
    ax[i][1].set_xlabel('Time')
    fig.colorbar(ax[i][1].get_images()[0], ax=ax[i][1], location='right')

# Differences plots (3rd column)
for i, predictions in enumerate(models_predictions):
    ax[i][2].imshow(test_data.T - predictions, aspect='auto', origin='lower', cmap='jet', extent=[0, T, 0, L])
    ax[i][2].set_title(model_names[i] + ' Difference')
    ax[i][2].set_ylabel('x')
    ax[i][2].set_xlabel('Time')
    fig.colorbar(ax[i][2].get_images()[0], ax=ax[i][2], location='right')

plt.tight_layout()
plt.savefig("img1.png")
plt.show()


