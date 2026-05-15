import numpy as np # linear algebra
from load_data import MnistDataloader


def init_params():
    # W1 = np.random.randn(128, 784) * np.sqrt(2.0 / 784)   # He init
    # b1 = np.zeros((128, 1))
    # W2 = np.random.randn(10, 128) * np.sqrt(2.0 / 128)
    # b2 = np.zeros((10, 1))

    params = np.load("params.npz")
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def deriv_ReLU(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2



def main():
    training_images_filepath = 'archive/train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = 'archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath)

    x_train, y_train = mnist_dataloader.read_images_labels(training_images_filepath, training_labels_filepath)

    W1, b1, W2, b2 = gradient_descent(x_train, y_train, 1000, 0.1)

    np.savez("params.npz", W1=W1, b1=b1, W2=W2, b2=b2)

main()