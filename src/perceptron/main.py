import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from Perceptron import Perceptron


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.make_blobs(
    n_samples=300, n_features=2, centers=3, cluster_std=1.05, random_state=4
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1, random_state=400
)

perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-perceptron.weights[0] * x0_1 -
        perceptron.bias) / perceptron.weights[1]
x1_2 = (-perceptron.weights[0] * x0_2 -
        perceptron.bias) / perceptron.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()
