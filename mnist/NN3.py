import time
import re
from random import random, sample, seed
from math import exp

start = time.time()


def relu(x):
    return max(0, x)


def drelu(rel):
    if rel >= 0:
        return 1
    return 0


def sigmoid(x):
    return 1 / (1 + exp(-x))


def dsigmoid(sig):
    return sig * (1 - sig)


class Neuron:
    def __init__(self, weights_number):
        self.weights = [random() - 0.5 for _ in range(weights_number + 1)]
        self.value = 0
        self.delta = 0

    def count_value(self, values, activation_function):
        self.value = 0
        for i in range(len(values)):
            self.value += values[i] * self.weights[i]
        self.value += self.weights[-1]
        if activation_function == 'relu':
            self.value = relu(self.value)
        elif activation_function == 'sigmoid':
            self.value = sigmoid(self.value)


class Layer:
    def __init__(self, neurons_this, neurons_previous, activation_function):
        self.neurons = [Neuron(neurons_previous) for _ in range(neurons_this)]
        self.activation_function = activation_function

    def forward(self, values):
        for neuron in self.neurons:
            neuron.count_value(values, self.activation_function)

    def get_values(self):
        values = []
        for neuron in self.neurons:
            values.append(neuron.value)
        return values


class NeuralNetwork:
    def __init__(self, hidden_layers_sizes, learning_rate):
        self.learning_rate = learning_rate
        self.layers = []
        self.layers.append(Layer(hidden_layers_sizes[0], 784, 'sigmoid'))
        # self.layers.append(Layer(hidden_layers_sizes[0], 2, 'relu'))
        for i in range(len(hidden_layers_sizes) - 1):
            self.layers.append(Layer(hidden_layers_sizes[i + 1], hidden_layers_sizes[i], 'sigmoid'))
        self.layers.append(Layer(10, hidden_layers_sizes[-1], 'sigmoid'))
        # self.layers.append(Layer(2, hidden_layers_sizes[-1], 'sigmoid'))

    def forward(self, input_image):
        self.layers[0].forward(input_image)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].forward(self.layers[i].get_values())

    def backward(self, input_image, label):
        previous_layer_values = self.layers[-2].get_values()
        for i in range(len(self.layers[-1].neurons)):
            current_neuron = self.layers[-1].neurons[i]
            expected_value = 0
            if i == label:
                expected_value = 1
            current_neuron.delta = (current_neuron.value - expected_value)
            if self.layers[-1].activation_function == 'sigmoid':
                current_neuron.delta *= dsigmoid(current_neuron.value)
            elif self.layers[-1].activation_function == 'relu':
                current_neuron.delta *= drelu(current_neuron.value)

            for j in range(len(previous_layer_values)):
                current_neuron.weights[j] -= current_neuron.delta * previous_layer_values[j] * self.learning_rate
            current_neuron.weights[-1] -= current_neuron.delta * self.learning_rate

        for i in range(len(self.layers) - 1):
            it = -(i + 2)
            if i == len(self.layers) - 2:
                previous_layer_values = input_image
            else:
                previous_layer_values = self.layers[it - 1].get_values()
            for k in range(len(self.layers[it].neurons)):
                current_neuron = self.layers[it].neurons[k]
                current_neuron.delta = 0
                for neuron in self.layers[it + 1].neurons:
                    current_neuron.delta += (neuron.weights[k] * neuron.delta)
                if self.layers[-1].activation_function == 'sigmoid':
                    current_neuron.delta *= dsigmoid(current_neuron.value)
                elif self.layers[-1].activation_function == 'relu':
                    current_neuron.delta *= drelu(current_neuron.value)
                for j in range(len(previous_layer_values)):
                    current_neuron.weights[j] -= current_neuron.delta * previous_layer_values[j] * self.learning_rate
                current_neuron.weights[-1] -= current_neuron.delta * self.learning_rate

    def predict(self, input_image):
        self.forward(input_image)
        values = self.layers[-1].get_values()
        return values.index(max(values))


def get_data(f):
    data = []
    for line in f:
        nums = re.findall(r'\b\d{1,3}(?:,\d{3})*(?!\d)', line)
        label = int(nums[0])
        nums.pop(0)
        for i in range(len(nums)):
            nums[i] = int(nums[i]) / 255
        data.append([nums, label])
    return data


def test():
    ok = 0
    for input, label in test_dataset:
        if label == N.predict(input):
            ok += 1
    print(ok / 100, '%')
    print('training time:', train_time, 'sec')


format_start = time.time()

trainf = open('train')
train_dataset = get_data(trainf)
trainf.close()
testf = open('test')
test_dataset = get_data(testf)
testf.close()

print('formating time:', time.time() - format_start, 'sec')

seed(17)
lr = 0.1
N = NeuralNetwork([256, 128], lr)
train_time = 0

for i in range(120):
    start_train_time = time.time()
    for input, label in sample(train_dataset, 500):
        N.forward(input)
        N.backward(input, label)
    train_time += (time.time() - start_train_time)
    print(i + 1, 'epochs done') # non SATORI
    # if i % 10 == 9:
        # test()

#for input, label in test_dataset: # SATORI
#    print(N.predict(input)) # SATORI

test() # non SATORI
print('all time:', (time.time() - start), 'sec') # non SATORI
# time ~ 1h20min, 94.77 %
