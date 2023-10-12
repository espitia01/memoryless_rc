import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx

def conv1d(input_sequence, kernel, stride=1):
    kernel_size = len(kernel)
    output_size = (len(input_sequence) - kernel_size) // stride + 1
    output = np.zeros(output_size)
    for i in range(0, output_size):
        start = i * stride
        end = start + kernel_size
        output[i] = np.dot(input_sequence[start:end], kernel)
    return output

def relu(x):
    return np.maximum(0, x)

def partition_and_pad(data, length):
    partitions = [data[i:i+length] for i in range(0, len(data), length)]
    return partitions

class CMRC:
    def __init__(self, dim_system, dim_reservoir, rho, sigma, density):
        self.dim_system = dim_system
        self.dim_reservoir = dim_reservoir
        self.r_state = np.zeros(dim_reservoir)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system) - .5)
        self.W_out = np.zeros((dim_system, dim_reservoir))
        self.conv1_kernel = np.random.randn(32)  # Updated filter size as per the appendix
        self.conv2_kernel = np.random.randn(1) 

    def forward(self, x):
        partition_length = 16
        padding = 8  # Updated padding size based on appendix
        x_padded = np.pad(x, (padding, padding), mode='wrap')
        partitions = partition_and_pad(x_padded, partition_length)
        result = []

        for partition in partitions:
            # First Convolution followed by ReLU activation
            y = conv1d(partition, self.conv1_kernel, stride=16)  # Stride value as described
            y = relu(y)

            # Square the reservoir states and concatenate
            y = np.hstack((y, y**2))

            # Second Convolution followed by ReLU activation
            y = conv1d(y, self.conv2_kernel, stride=1)  # Stride of 1 for second convolution
            y = relu(y)

            # Update reservoir state
            self.advance_r_state(y)

            # Get the output from reservoir state
            result.append(self.v())
        return np.concatenate(result)

    def advance_r_state(self, u):
        self.r_state = self.sigmoid(np.dot(self.W_in, u))
        return self.r_state
    
    def set_state(self, state):
        self.r_state = state

    def v(self):
        return np.dot(self.W_out, self.r_state)

    def train(self, trajectory):
        R = np.zeros((self.dim_reservoir, trajectory.shape[0]))
        for i in range(trajectory.shape[0]):
            R[:, i] = self.r_state
            u = trajectory[i]
            self.advance_r_state(u)
        self.W_out = self.linear_regression(R, trajectory)

    def predict(self, steps):
        prediction = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.v()
            prediction[i] = v
            self.advance_r_state(prediction[i])
        return prediction

    def sigmoid(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def linear_regression(self, R, trajectory, beta=0.0001):
        Rt = np.transpose(R)
        inverse_part = np.linalg.inv(np.dot(R, Rt) + beta * np.identity(R.shape[0]))
        return np.dot(np.dot(trajectory.T, Rt), inverse_part)