import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx

class ReservoirComputer:
    def __init__(self, dim_system, dim_reservoir, rho, sigma, density):
        self.dim_system = dim_system
        self.dim_reservoir = dim_reservoir
        self.r_state = np.zeros(dim_reservoir)
        self.A = generate_adjacency_matrix(dim_reservoir, rho, sigma, density)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system) - .5)
        self.W_out = np.zeros((dim_system, dim_reservoir))

    def advance_r_state(self, u):
        self.r_state = sigmoid(np.dot(self.A, self.r_state) + np.dot(self.W_in, u))
        return self.r_state

    def v(self):
        return np.dot(self.W_out.T, self.r_state)  # Note the transpose (.T) on self.W_out


    def set_state(self, state):
        self.r_state = state

    def train(self, trajectory):
        #trajectory has shape (n, dim_system), n is the number of timesteps
        R = np.zeros((self.dim_reservoir, trajectory.shape[0]))
        for i in range(trajectory.shape[0]):
            R[:, i] = self.r_state
            u = trajectory[i]
            self.advance_r_state(u)
        self.W_out = linear_regression(R, trajectory).T

    def predict(self, steps):
        prediction = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.v()
            prediction[i] = v
            self.advance_r_state(v)
        return prediction


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def generate_adjacency_matrix(dim_reservoir, rho, sigma, density):
    graph = nx.erdos_renyi_graph(dim_reservoir, density)
    graph = nx.to_numpy_array(graph)

    random_array = 2 * sigma * (np.random.rand(dim_reservoir, dim_reservoir) - 0.5)
    rescaled = graph * random_array
    return scale_matrix(rescaled, rho)


def scale_matrix(A, rho):
    eigenvalues, _ = np.linalg.eig(A)
    max_eigenvalue = np.amax(eigenvalues)
    A = A/np.absolute(max_eigenvalue) * rho
    return A

def linear_regression(R, trajectory, beta=0.0001):
    Rt = np.transpose(R)
    inverse_part = np.linalg.inv(np.dot(R, Rt) + beta * np.identity(R.shape[0]))
    return np.dot(np.dot(trajectory.T, Rt), inverse_part)
