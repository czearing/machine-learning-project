import math


def tanh(x):
    """ Smooth activation function that is symmetric around the origin."""
    return (2/(1 + math.exp(-2*x))) - 1
