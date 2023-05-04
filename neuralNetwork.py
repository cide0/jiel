import numpy
import scipy.special
import matplotlib.pyplot


class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lRate = learningRate

    def train(self):
        print('I am training')

    def query(self):
        print('I ask..')


if __name__ == '__main__':
    inputNodes = 3
    hiddenNodes = 3
    outputNodes = 3
    learningRate = 0.5
    neuralNetwork = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    neuralNetwork.train()
    neuralNetwork.query()
