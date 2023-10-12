import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx

def ks_pseudospectral(u, t, L, N):
    x = np.linspace(0, L, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)

    uhat = np.fft.fft(u)
    uhat_der = 1j * k * uhat
    u_der = np.real(np.fft.ifft(uhat_der))

    uhat_der2 = -k**2 * uhat
    u_der2 = np.real(np.fft.ifft(uhat_der2))

    uhat_der4 = k**4 * uhat
    u_der4 = np.real(np.fft.ifft(uhat_der4))

    return -u * u_der - u_der2 - u_der4