# Testing an implementation of a basic neural network I found online 
import numpy as np
import csv 


######################################################################

#CLASSES AND FUNCTIONS

def sigmoid(x): 
    # sigmoid funcction: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x)) 

class Neuron: 
    def __init__(self, weights, bias): 
        self.weights = weights
        self.bias = bias 

    def feedforward(self, inputs): 
        # weight inputs, add bias, then use activation function
        # dot product is np.dot 
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total) 

class NeuralNetwork_one: 
    def __init__(self): 
        weights = np.array([0,1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x): 
        out_h1 = self.h1.feedforward(x) 
        out_h2 = self.h2.feedforward(x) 
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

def mse_loss(y_true, y_pred): 
    return ((y_true -y_pred)**2).mean()

###################################################################

# MAIN CODE

# Note: When creating strings for a file name, place r in front to change it into a raw string
# Python has trouble understand it if it is a normal string, which is without the r.

#path = 'C:\\Users\\nnguy\\cs_docs\\github_repos\\Misc_Projects\\python\\test_data\\neural.csv'
#path = r'C:\Users\nnguy\cs_docs\github_repos\Misc_Projects\python\test_data\neural.csv'
path = r'test_data\neural.csv'

with open(path, 'r') as csvfile:
    data = list(csv.reader(csvfile))

# 2D array = array[row][column]
# convering to easier to compute data
for i in range(4): 
    data[i+1][1] = int(data[i+1][1]) - 135
    data[i+1][2] = int(data[i+1][2]) - 66
    if (data[i+1][3] == 'M'):
        data[i+1][3] = '0'
    else: 
        data[i+1][3] = '1' 
    data[i+1][3] = int(data[i+1][3])

print(data)

#left out on #4 of website