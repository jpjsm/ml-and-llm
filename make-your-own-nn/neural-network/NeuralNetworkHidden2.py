import numpy
import scipy.special


class NeuralNetworkHidden2:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.whh = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.hnodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # Query
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Hidden layer N° 1
        hidden1_inputs = numpy.dot(self.wih, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # Hidden layer N° 2
        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = numpy.dot(self.who, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)


        # backpropagation
        output_errors = targets - final_outputs
        hidden2_errors = numpy.dot(self.who.T, output_errors)
        hidden1_errors = numpy.dot(self.whh.T, hidden2_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden2_outputs))
        self.whh += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1 - hidden2_outputs)), numpy.transpose(hidden1_outputs))
        self.wih += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1 - hidden1_outputs)), numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        # Hidden layer N° 1
        hidden1_inputs = numpy.dot(self.wih, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # Hidden layer N° 2
        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = numpy.dot(self.who, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
