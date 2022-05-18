import math


def sigmoid(x):
    """ Sigmoid activation function. The activation curve is more smooth but can lead to a vanishing gradient."""
    return 1/(1+math.exp(-x))
