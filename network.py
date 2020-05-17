import numpy as np
import random

class Network:
	def __init__(self, layer_sizes):
		"""Constructs a neural network from layer_sizes, a list containing
		the number of neurons in each layer.

		Weights are Gaussian distributed with a mean of 0 and a
		standard deviation of 1/sqrt(number of input neurons), biases have mean
		0 and standard deviation 1"""

		self.layers = len(layer_sizes)
		self.sizes = layer_sizes
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x)
				  for x,y in zip(self.sizes[:-1], self.sizes[1:])]

	def train(self, training_data, epochs, mb_size, eta,
		   lmbda = 0.0,
		   eval_data = None,
		   monitor_eval_cost = False,
		   monitor_eval_accuracy = False,
		   monitor_training_cost = False,
		   monitor_training_accuracy = False):
		"""Trains the neural network using stochastic gradient descent.
		Evaluation data is used to modify the hyper parameters and prevent
		overfitting.

		training_data = tuple containing all training examples
		epochs = number of iterations to train for
	    mb_size = mini batch size
		eta = learning rate,
		lmbda = regularization paramter
		eval_data = tuple containing all evaluation examples
		
		"""
		if eval_data:
			eval_len = len(eval_data)
		n = len(training_data)
		eval_cost, eval_accuracy = [], []
		for e in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mb_size]
				   for k in range(0, n, mb_size)]
			for mb in mini_batches:
				self.train_mini_batch(mb, eta, lmbda, n)
			print("Finished epoch {}.".format(e))
			# print information if enabled
			if monitor_training_cost:
				cost = self.total_cost(training_data, lmbda)
				training_cost.append(cost)
				print("Training data cost: {}".format(cost))
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data)
				training_accuracy.append(accuracy)
				print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
			if monitor_eval_cost:
				cost = self.total_cost(eval_data, lmbda)
				evaluation_cost.append(cost)
				print("Cost on evaluation data: {}".format(cost))
			if monitor_eval_accuracy:
				accuracy = self.accuracy(eval_data)
				eval_accuracy.append(accuracy)
				print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(eval_data), n))

		return (eval_cost, eval_accuracy, \
            training_cost, training_accuracy)

	def train_mini_batch(self, mb, eta, lmbda, n):
		"""Update weights and biases based on the average cost of mini-batch
		mb.
		mb = mini batch
		eta = learning rate
		lmbda = regularization parameter
		n = number of all training examples

		The mini batch is a tuple containing (img, label), where
		img contains the image_data in a np array, and label is
		a 10-dimensional np array where the answer is set
		"""
		# sum of gradients of every example in the mini batch
		b_gradient = [np.zeros(b.shape) for b in self.biases]
		w_gradient = [np.zeros(w.shape) for w in self.weights]
		for x in mb:
			# backpropagate to find the gradient in this mini batch
			db_gradient, dw_gradient = self.backprop(x[0], x[1])
			# add this gradient to a running total
			b_gradient = [nb+dnb for nb, dnb in zip(b_gradient, db_gradient)]
			w_gradient = [nw+dnw for nw, dnw in zip(w_gradient, dw_gradient)]
		# update the weights and biases based off the average gradient, the
		# learning rate, and lmbda
		m = len(mb)
		self.weights = [(1-eta*(lmbda/n))*w - (eta/m)*wg
			for w, wg in zip(self.weights, w_gradient)]
		self.biases = [b-(eta/m)*bg
			for b, bg in zip(self.biases, b_gradient)]

	def backprop(self, img, label):
		"""Return a tuple containing the bias gradient and weight gradient
		with resepct to the cost. Backpropagates to find the error in each
		neuron, which is used to compute the partial derivative of the weight
		and bias with respect to the cost.

		"""
		# bias gradient is a matrix with the same shape as self.biases
		b_gradient = [np.zeros(b.shape) for b in self.biases]
		# weight gradient is a vector of matrices with same shape as weights
		w_gradient = [np.zeros(w.shape) for w in self.weights]

		# compute the output of the neural network, keeping track of the
		# activations and weighted sums (zs) in each layer
		# the first layer of neurons takes the image data as input
		a = img
		activations = [img] # vector of each activation vector
		zs = [] # vector of each weighted sum vector
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, a)+b
			zs.append(z)
			a = sigmoid(z)
			activations.append(a)
		# compute the error of the last layer, which is a gradient of
		# the partial derivatives of the weighted sums wrt cost
		error = dz_wrt_cost(activations[-1], label)
		# part derivative of bias IS the error, while part derivative of
		# weight is the dot product between the error and previous activation
		# layer (tranposed)
		b_gradient[-1] = error
		w_gradient[-1] = np.dot(error, activations[-2].transpose())
		# backpropagate to find the error of each layer
		for l in range(2, self.layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			# calculate error from the error in the next layer
			error = np.dot(self.weights[-l+1].transpose(), error) * sp
			b_gradient[-l] = error
			w_gradient[-l] = np.dot(error, activations[-l-1].transpose())
		return (b_gradient, w_gradient)

	def feedforward(self, a):
		"""Return output given input activations a"""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

	# functions for data monitoring
	def total_cost(self, data, lmbda):
		"""Computes the total cost of the network in a collection of
		training examples (data)

		"""
		m = len(data)
		c = 0.0
		for x in data:
			a = self.feedforward(x[0])
			# divide by the # of examples because we're taking average
			c += self.cost(a, x[1])/m
		# L2 regularization
		cost += 0.5*(lmbda/m)*sum(
			np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def accuracy(self, data):
		""" Returns the amount of digits that the network can correctly
		identify. The network's answer is taken to be the neuron with
		the highest activation in the last layer.

		"""
		results = [(np.argmax(self.feedforward(x[0])), np.argmax(x[1]))
			for x in data]

		return sum(int(x == y) for (x, y) in results)

# calculating functions
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def dz_wrt_cost(a, y):
	return (a-y)

def cost(a, y):
	""" Cross-entropy cost function """
	return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

if __name__ == "__main__":
	import mnist_loader as ml
	training_data, eval_data, test_data = ml.load_all()
	net = Network([784, 30, 10])
	net.train(training_data, 30, 10, 0.5, 5.0, eval_data, 
		   monitor_eval_accuracy=True)
