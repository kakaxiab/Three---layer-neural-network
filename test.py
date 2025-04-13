import numpy as np


def test(model, X_test, y_test):
    scores = model.forward(X_test)
    pred = np.argmax(scores, axis=1)
    accuracy = np.mean(pred == y_test)
    print(f'Test accuracy: {accuracy}')
    return accuracy
    