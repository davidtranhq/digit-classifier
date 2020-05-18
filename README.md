# Digit Recognizer

A Python 3 implementation of a feed-forward neural network trained with backpropagation and stochastic gradient descent to recognize handwritten digits. The algorithm is outlined in Michael Nielsen's book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). Includes a GUI.

The training and testing examples are taken from the [MNSIT handwritten digit database](http://yann.lecun.com/exdb/mnist/).

## Getting Started

### Dependencies

* [Python 3](https://www.python.org/downloads/)
* tkinter (only required for the GUI, install with `pip install tkinter`)
* Pillow (only required for the GUI, install with `pip install pillow`)

### Installation

Clone the repository with:

`git clone https://github.com/davidtranhq/digit-classifier`

Alternatively, download a standalone Windows 10 executable [here]()

## Usage

To use a Python module:

Example (no gui):
```
>>> import mnist_loader
>>> import network
>>> training_data, eval_data, test_data = mnist_loader.load_all()
>>> layers = [784, 100, 10]
>>> net = Network(layers)
>>> net.train(training_data, 60, 10, 0.1, 5.0, eval_data)
>>> correct_answers = net.accuracy(test_data)
```

or

```
>>> python network.py
```

Example (using gui):
```
>>> python gui.py
```

## Details








	



