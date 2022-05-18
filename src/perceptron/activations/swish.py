import math


def swish(x):
    """ Activation function similar to ReLu but is differentiable at all points."""
    return x/(1-math.exp(-x))
