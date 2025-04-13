import numpy as np
from model import ThreeLayerNeuralNetwork
from train import train


def hyperparameter_search(X_train, y_train, X_val, y_val):
    learning_rates = [1e-1, 1e-2, 1e-5]
    hidden_sizes = [100, 300, 500]
    reg_strengths = [1e-6, 1e-3, 1e-2]

    best_val = -1
    best_hyperparams = {}

    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in reg_strengths:
                model = ThreeLayerNeuralNetwork(X_train.shape[1], hs, 10)
                trained_model, _, _, _ = train(model, X_train, y_train, X_val, y_val,
                                               learning_rate=lr, reg=reg, num_iters=500)
                print(type(trained_model))  
                val_scores = trained_model.forward(X_val)
                val_pred = np.argmax(val_scores, axis=1)
                val_acc = np.mean(val_pred == np.argmax(y_val))  # 确保标签处理一致

                if val_acc > best_val:
                    best_val = val_acc
                    best_hyperparams = {
                        'learning_rate': lr,
                        'hidden_size': hs,
                        'reg': reg
                    }

    print(f'Best validation accuracy: {best_val}')
    print(f'Best hyperparameters: {best_hyperparams}')
    return best_hyperparams