def leakyReLu(x, axisWeight=0.01):
    """Returns the axisWeight (defaults to 0.01) multiplied by the input if it is negative, otherwise it returns the input."""
    if x < 0:
        return axisWeight * x
    else:
        return x
