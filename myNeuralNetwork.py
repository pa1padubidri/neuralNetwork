from numpy import exp, array, random, dot

class NeuralNetwork():
	"""Single layer neural network"""
	def __init__(self):
		random.seed(1)
		self.synaptic_weights =2* random.random((3,1)) - 1

	def __sigmoid(self, x, derivative=False):
		if derivative:
			return x*(1- x)
		return 1/(1+exp(-x))
		
	def train(self, training_inputs, training_outputs, training_iterations):
		for iteration in xrange(training_iterations):
			predicted_outputs = self.predict(training_inputs)
			error = training_outputs - predicted_outputs
			self.synaptic_weights += dot(training_inputs.T, error* self.__sigmoid(predicted_outputs, derivative=True))

	def predict(self, training_inputs):
		return self.__sigmoid(dot(training_inputs, self.synaptic_weights))

if __name__ == "__main__":

	neural_network = NeuralNetwork()
	print "Initial Weights: {}".format(neural_network.synaptic_weights)

	training_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_outputs = array([[0,1,1,0]]).T

	neural_network.train(training_inputs, training_outputs, 1000)

	print "Training Complete\nFinal Weights: {}".format(neural_network.synaptic_weights)

	print "Prediction for Test input [1,0,0]: {}".format(neural_network.predict([1,0,0]))