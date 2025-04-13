import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
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

def show_image_samples(images, labels, class_names, num_samples=10):
    """显示图像样本"""
    plt.figure(figsize=(10, 5))
    indices = np.random.choice(len(images), num_samples)
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        # CIFAR-10图像格式为(3072,)，前1024是R，中间1024是G，最后1024是B
        img = images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(class_names[labels[idx]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 加载数据
    train_data, train_labels, test_data, test_labels = load_cifar10('cifar-10-batches-py')

    # 打印数据集基本信息
    print("\n=== 数据集基本信息 ===")
    print(f"训练集形状: {train_data.shape} (样本数, 特征数)")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"测试集标签形状: {test_labels.shape}")
    print(f"像素值范围: {np.min(train_data)}-{np.max(train_data)}")

    # 打印类别分布
    print("\n=== 类别分布 ===")
    print("训练集:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"{class_names[cls]}: {count} ({count/len(train_labels):.2%})")
    
    print("\n测试集:")
    unique, counts = np.unique(test_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"{class_names[cls]}: {count} ({count/len(test_labels):.2%})")

    # 显示样本图像
    print("\n显示训练集样本示例...")
    show_image_samples(train_data, train_labels, class_names)
    
    print("\n显示测试集样本示例...")
    show_image_samples(test_data, test_labels, class_names)

    # 划分验证集
    num_val = 1000
    X_val, y_val = train_data[:num_val], train_labels[:num_val]
    X_train, y_train = train_data[num_val:], train_labels[num_val:]

    # 打印预处理信息
    print("\n=== 预处理信息 ===")
    print(f"训练样本数: {len(X_train)}")
    print(f"验证样本数: {len(X_val)}")
    print(f"测试样本数: {len(test_data)}")
    print("将进行归一化: 像素值/255.0")

    # 初始化模型
    model = ThreeLayerNeuralNetwork(
        input_size=3072,
        hidden_size=512,
        output_size=10,
        activation='relu'
    )

    # 打印模型信息
    print("\n=== 模型架构 ===")
    print(f"输入层: {model.input_size} 个神经元 (32x32x3)")
    print(f"隐藏层: {model.hidden_size} 个神经元")
    print(f"输出层: {model.output_size} 个神经元")
    print(f"激活函数: {model.activation}")
    
   
    
def analyze_class_distribution(images, labels, class_names):
    """对不同类别进行描述性统计分析"""
    print("\n=== 类别描述性统计 ===")
    
    # 1. 各类别样本数量统计
    unique, counts = np.unique(labels, return_counts=True)
    print("\n1. 各类别样本数量:")
    for cls, count in zip(unique, counts):
        print(f"{class_names[cls]:<10}: {count:>5} 张 ({count/len(labels):.2%})")
    
    # 2. 像素值统计（按类别）
    print("\n2. 各类别像素均值统计:")
    print(f"{'类别':<10} {'R均值':<8} {'G均值':<8} {'B均值':<8}")
    for cls in range(len(class_names)):
        class_images = images[labels == cls]
        if len(class_images) > 0:
            # 计算每个通道的均值
            r_mean = np.mean(class_images[:, :1024])  # 前1024是R通道
            g_mean = np.mean(class_images[:, 1024:2048])  # 中间1024是G通道
            b_mean = np.mean(class_images[:, 2048:])  # 最后1024是B通道
            print(f"{class_names[cls]:<10} {r_mean:<8.1f} {g_mean:<8.1f} {b_mean:<8.1f}")
    
    # 3. 像素值分布可视化（按类别）
    plt.figure(figsize=(12, 6))
    for i, cls in enumerate(range(len(class_names))):
        if np.sum(labels == cls) == 0:
            continue
            
        class_images = images[labels == cls]
        # 随机选择一张图像
        sample_img = class_images[np.random.randint(0, len(class_images))]
        sample_img = sample_img.reshape(3, 32, 32).transpose(1, 2, 0)
        
        plt.subplot(2, 5, i+1)
        plt.imshow(sample_img)
        plt.title(f"{class_names[cls]}\n(n={np.sum(labels == cls)})")
        plt.axis('off')
    plt.suptitle("各类别示例图像及样本数量")
    plt.tight_layout()
    plt.show()
    
    # 4. 各类别像素强度分布
    plt.figure(figsize=(12, 6))
    colors = ['red', 'green', 'blue']
    for cls in range(len(class_names)):
        if np.sum(labels == cls) == 0:
            continue
            
        class_images = images[labels == cls]
        # 计算每个通道的像素强度分布
        for ch in range(3):
            channel_data = class_images[:, ch*1024:(ch+1)*1024].flatten()
            plt.hist(channel_data, bins=50, alpha=0.5, 
                     color=colors[ch], label=f'{class_names[cls]} {["R","G","B"][ch]}')
    
    plt.title("各类别RGB通道像素强度分布")
    plt.xlabel("像素值")
    plt.ylabel("频数")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 在main函数中添加调用
if __name__ == "__main__":
    # ... (之前的代码保持不变)
    
    # 添加类别分析
    print("\n=== 训练集类别分析 ===")
    analyze_class_distribution(train_data, train_labels, class_names)
    
    print("\n=== 测试集类别分析 ===")
    analyze_class_distribution(test_data, test_labels, class_names)
    
    # ... (之后的代码保持不变)