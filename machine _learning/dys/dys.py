import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        # 初始化网络参数（权重和偏置）
        self.learning_rate = learning_rate
        self.layers = []
        self.biases = []

        # 初始化每一层的权重和偏置
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        # 存储每一层的中间结果
        self.zs = []  # 线性变换结果
        self.activations = [x]  # 激活后的结果

        for i in range(len(self.layers)):
            z = np.dot(self.activations[-1], self.layers[i]) + self.biases[i]
            self.zs.append(z)
            if i == len(self.layers) - 1:  # 输出层使用softmax
                activation = self.softmax(z)
            else:  # 隐藏层使用ReLU
                activation = self.relu(z)
            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, x, y):
        # 前向传播获取预测值
        output = self.forward(x)

        # 计算输出层的误差
        m = y.shape[0]  # 样本数
        dz = output - y  # 输出层误差（交叉熵损失的梯度）

        for i in reversed(range(len(self.layers))):
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                da = np.dot(dz, self.layers[i].T)
                dz = da * self.relu_derivative(self.zs[i - 1])

            # 更新权重和偏置
            self.layers[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train_one_sample(self, x, y):
        # 将输入 x 和 y 转换为行向量形式
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.backward(x, y)

    def predict_prob(self, x):
        return self.forward(x)

    def predict(self, x):
        probs = self.predict_prob(x)
        return np.argmax(probs, axis=1)


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 标签

# 数据预处理
# 将标签转换为独热编码
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 创建 MLP 模型
input_size = X_train.shape[1]  # 特征维度
hidden_sizes = [10, 8]  # 两个隐藏层，大小分别为 10 和 8
output_size = y_train.shape[1]  # 类别数
mlp = MLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, learning_rate=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    for i in range(X_train.shape[0]):
        mlp.train_one_sample(X_train[i], y_train[i])
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} completed")

# 测试模型
y_test_probs = mlp.predict_prob(X_test)
y_test_preds = mlp.predict(X_test)

# 计算测试集准确率
y_test_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(y_test_preds == y_test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
