import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, X_train, y_train, X_val, y_val, 
          learning_rate=1e-3, lr_decay=0.95, 
          reg=1e-5, epochs=100, batch_size=256,
          early_stop=5, verbose=True):
    
    # 数据预处理（使用model.output_size）（关键修改点2）
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    y_train_onehot = np.eye(model.output_size)[y_train]  # 使用新增的属性
    y_val_onehot = np.eye(model.output_size)[y_val]     # 使用新增的属性

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0
    best_params = None
    no_improve = 0
    
    for epoch in tqdm(range(epochs), desc="Training"):
        if epoch % 10 == 0 and epoch > 0:
            learning_rate *= lr_decay
        
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch, y_batch = X_train[batch_idx], y_train_onehot[batch_idx]
            
            # 数据增强
            X_batch = random_flip(X_batch)
            
            model.forward(X_batch)
            grads = model.backward(X_batch, y_batch, reg)
            
            for param in model.params:
                model.params[param] -= learning_rate * grads[param]
        
        # 评估
        train_loss = model.compute_loss(X_train, y_train_onehot, reg)
        train_pred = model.forward(X_train).argmax(axis=1)
        train_acc = (train_pred == y_train).mean()
        
        val_loss = model.compute_loss(X_val, y_val_onehot, reg)
        val_pred = model.forward(X_val).argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
            
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, lr={learning_rate:.2e}")
    
    model.params = best_params
    
    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, history

def random_flip(X, p=0.5):
    """随机水平翻转图像"""
    if len(X.shape) == 2:  # 如果数据是展平的
        # 将展平的数据reshape回32x32x3格式
        X_reshaped = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        flip_mask = np.random.rand(len(X_reshaped)) < p
        X_reshaped[flip_mask] = X_reshaped[flip_mask, :, ::-1, :]  # 水平翻转
        # 重新展平数据
        return X_reshaped.transpose(0, 3, 1, 2).reshape(-1, 3072)
    else:
        # 如果数据已经是图像格式
        flip_mask = np.random.rand(len(X)) < p
        X[flip_mask] = X[flip_mask, :, ::-1, :]
        return X