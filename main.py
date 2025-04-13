import numpy as np
import pickle
import os
from model import ThreeLayerNeuralNetwork
from train import train

def load_cifar10(data_dir):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # 加载训练集
    train_data = []
    train_labels = []
    for i in range(1, 6):
        file = os.path.join(data_dir, f'data_batch_{i}')
        batch = unpickle(file)
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    train_data = np.concatenate(train_data)
    train_labels = np.array(train_labels)

    # 加载测试集
    test_file = os.path.join(data_dir, 'test_batch')
    test_batch = unpickle(test_file)
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":
    # 加载数据
    train_data, train_labels, test_data, test_labels = load_cifar10('cifar-10-batches-py')

    # 划分验证集
    num_val = 1000
    X_val, y_val = train_data[:num_val], train_labels[:num_val]
    X_train, y_train = train_data[num_val:], train_labels[num_val:]

    # 初始化模型（确保传递output_size）（关键修改点3）
    model = ThreeLayerNeuralNetwork(
        input_size=3072,  # 32x32x3
        hidden_size=512,
        output_size=10,    # 必须明确指定
        activation='relu'
    )

    # 训练
    trained_model, history = train(
        model,
        X_train, y_train,
        X_val, y_val,
        learning_rate=1e-3,
        lr_decay=0.95,
        reg=1e-4,
        epochs=200,
        batch_size=256,
        early_stop=10
    )

    # 测试评估
    test_scores = trained_model.forward(test_data.astype('float32')/255)
    test_pred = np.argmax(test_scores, axis=1)
    test_acc = np.mean(test_pred == test_labels)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # 保存模型
    with open('best_model.pkl', 'wb') as f:
        pickle.dump({
            'params': trained_model.params,
            'history': history
        }, f)