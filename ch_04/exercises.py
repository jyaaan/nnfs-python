# type: ignore
""" ReLU Implementation
"""
import numpy as np
from nnfs.datasets import spiral_data
import nnfs



inputs = [
    0,
    2,
    -1,
    3.3,
    -2.7,
    1.1,
    2.2,
    -100,
]

output = []
for i in inputs:
    output.append(max(0, i))

print(f'{output=}')

# now with numpy

output = np.maximum(0, inputs)
print(f'{output=}')


nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases



class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense1.forward(X)

activation1.forward(dense1.output)

print(f'{activation1.output[:5]=}')

# let's