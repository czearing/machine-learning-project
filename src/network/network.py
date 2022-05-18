class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # Adds a new layer to the Network
    def add(self, layer):
        self.layers.append(layer)

    # Sets loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Predicts an output for the given data
    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # Trains the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            loss = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                loss += self.loss(y_train[j], output)
                newError = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    newError = layer.backward_propagation(
                        newError, learning_rate)
            loss /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, loss))
