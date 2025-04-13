import numpy as np

class ThreeLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        # 新增：保存output_size作为类属性（关键修改点1）
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  # 新增属性
        self.activation = activation

        # He初始化
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size),
            'b1': np.zeros(hidden_size),  # 修改为1D数组
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size),
            'b2': np.zeros(output_size)   # 修改为1D数组
        }
        self.cache = {}  # 显式初始化cache

    def forward(self, X):
        self.cache = {'X': X}  # 缓存输入数据
        
        # 第一层
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        if self.activation == 'relu':
            A1 = np.maximum(0, Z1)
        elif self.activation == 'sigmoid':
            A1 = 1 / (1 + np.exp(-Z1))
        self.cache.update({'Z1': Z1, 'A1': A1})

        # 第二层
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        exp_scores = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.cache.update({'Z2': Z2, 'A2': A2})
        return A2

    def backward(self, X, y, reg_lambda=0.0):
        m = X.shape[0]
        A1, A2 = self.cache['A1'], self.cache['A2']
        W1, W2 = self.params['W1'], self.params['W2']

        # 输出层梯度
        dZ2 = A2 - y
        grads = {
            'W2': A1.T.dot(dZ2)/m + reg_lambda*W2,
            'b2': np.sum(dZ2, axis=0)/m
        }

        # 隐藏层梯度
        dA1 = dZ2.dot(W2.T)
        if self.activation == 'relu':
            dZ1 = dA1 * (A1 > 0)
        elif self.activation == 'sigmoid':
            dZ1 = dA1 * A1 * (1 - A1)
        
        grads.update({
            'W1': X.T.dot(dZ1)/m + reg_lambda*W1,
            'b1': np.sum(dZ1, axis=0)/m
        })
        return grads

    def compute_loss(self, X, y, reg_lambda=0.0):
        m = y.shape[0]
        A2 = self.forward(X)
        correct_logprobs = -np.log(A2[np.arange(m), y.argmax(axis=1)] + 1e-8)
        data_loss = np.sum(correct_logprobs)/m
        reg_loss = 0.5*reg_lambda*(np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return data_loss + reg_loss