import math


def ELU(x, axisWeight=0.01):
    """A leaky ReLu activation function with a logarithmic curve for negative values."""
    if x < 0:
        return axisWeight * (math.exp(x)-1)
    else:
        return x
