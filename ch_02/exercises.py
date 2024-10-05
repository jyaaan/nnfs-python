# type: ignore
inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]
biases = [2, 3, 0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0.0

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight

    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(f"{layer_outputs=}")

# Do it with numpy for a single neuron
import numpy as np

outputs = np.dot(weights[0], inputs) + biases[0]

print(f"{outputs=}")

# now do it for the entire layer
print(f"{np.dot(weights, inputs)=}")
layer_outputs = np.dot(weights, inputs) + biases
print(f"{layer_outputs=}")

# a row vector with 3 values in numpy
np.array([[1, 2, 3]])  # note double brackets
# equivalent to
a = [1, 2, 3]
a_row = np.array([a])

# which is also equivalent to
np.expand_dims(np.array(a), axis=0)

# transposing a list into a row vector into a column vector
b = [2, 3, 4]
b_col = np.array([b]).T

np.dot(a_row, b_col)  # >>> array([[20]])

# performing matrix multiplication on a sample set and 3 neurons
inputs = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(f"{layer_outputs=}")
