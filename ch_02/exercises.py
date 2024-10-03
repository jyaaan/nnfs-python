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
