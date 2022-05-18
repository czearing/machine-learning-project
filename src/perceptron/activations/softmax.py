import math


def softmax(x):
    """A combination of multiple sigmoids. Especially helpful for multi-classification problems."""
    z = math.exp(x)
    z_ = z/z.sum()
    return z_
