import numpy as np # linear algebra
from load_data import MnistDataloader


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


def test_all(X, Y, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    predictions = np.argmax(A2, axis=0)
    accuracy = np.mean(predictions == Y)
    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * Y.size)}/{Y.size})")


def test_one(data, actual, W1, b1, W2, b2):
    Z1 = W1.dot(data) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    prediction = int(np.argmax(A2, axis=0)[0])
    confidence = float(A2[prediction, 0])
    correct = "correct" if prediction == actual else "wrong"
    print(f"Predicted: {prediction} (confidence {confidence:.2%}) | Actual: {actual} | {correct}")

def main():
    test_images_filepath = 'archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = 'archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    mnist_dataloader = MnistDataloader(test_images_filepath, test_labels_filepath)

    x_test, y_test = mnist_dataloader.read_images_labels(test_images_filepath, test_labels_filepath)

    params = np.load("params.npz")
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']


    for i in range(200, 300):
        test_one(x_test[:, i:i+1], y_test[i], W1, b1, W2, b2)

main()