import numpy
import scipy.special
import matplotlib.pyplot


class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lRate = learningRate
        self.wih = (numpy.random.rand(self.hNodes, self.iNodes) - 0.5)
        self.who = (numpy.random.rand(self.oNodes, self.hNodes) - 0.5)
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        print("I startet training ...")
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Some debuginformations
        print("Final output:")
        print(final_outputs)
        pass

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    inputNodes = 3
    hiddenNodes = 3
    outputNodes = 3
    learningRate = 0.5
    neuralNetwork = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    input_list = [(1.0, 0.5, 0.3), (0.4, 0.9, 0.1)]
    target_list = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    neuralNetwork.train(input_list, target_list)

    output = neuralNetwork.query([1.0, 0.5, -1.5])
    print("Queryoutput:")
    print(output)
