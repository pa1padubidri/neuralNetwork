from numpy import exp, dot, array, random

class NeuronLayer():
	"""NeuronLayer initialises layers"""
	def __init__(self, neurons, inputs_per_neurons):
		random.seed(1)
		self.synaptic_weights =2 * random.random((inputs_per_neurons, neurons)) - 1

class NeuralNetwork():
	"""docstring for NeuralNetwork"""
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2

	def __sigmoid(self, x, derivative=False):
		if derivative:
			return x*(1- x)
		return 1/(1+exp(-x))

	def train(self, training_inputs, training_outputs, training_iterations):
		for iteration in xrange(training_iterations):
			l1_outputs, l2_outputs = self.predict(training_inputs)

			l2_error = training_outputs - l2_outputs
			l2_delta = l2_error * self.__sigmoid(l2_outputs, derivative=True)

			l1_error = dot(l2_delta, self.layer2.synaptic_weights.T)
			l1_delta = l1_error * self.__sigmoid(l1_outputs)

			self.layer1.synaptic_weights += dot(training_inputs.T, l1_delta)
			self.layer2.synaptic_weights += dot(l1_outputs.T, l2_delta)

	
	def predict(self, training_inputs):
		l1_outputs = self.__sigmoid(dot(training_inputs, self.layer1.synaptic_weights))
		l2_outputs = self.__sigmoid(dot(l1_outputs, self.layer2.synaptic_weights))
		return l1_outputs, l2_outputs

	def printWeights(self):
		print "After Training: "
		print "Layer1: ", self.layer1.synaptic_weights
		print "Layer2: ", self.layer2.synaptic_weights


if __name__ == "__main__":
	
	layer1 = NeuronLayer(2,3)
	layer2 = NeuronLayer(1,2)

	print "Before Training: "
	print "	Layer1: \n", layer1.synaptic_weights
	print "	Layer2: \n", layer2.synaptic_weights

	neural_network = NeuralNetwork(layer1, layer2)

	training_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
	training_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

	neural_network.train(training_inputs, training_outputs,60000)

	neural_network.printWeights()

	hidden_layer, output = neural_network.predict([0,1,0])
	print "Prediction for Test input [0,1,0]: {}".format(output)