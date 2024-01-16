import numpy as np


# 定义激活函数 sigmoid
def sigmoid(x):
    """
    sigmoid激活函数

    参数:
    x -- 输入

    返回值:
    返回sigmoid函数的运算结果
    """
    return 1 / (1 + np.exp(-x))


# 定义神经网络类
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        神经网络类构造函数

        参数:
        input_size -- 输入层大小
        hidden_size -- 隐藏层大小
        output_size -- 输出层大小
        """

        # 初始化权重和偏置
        #形状为(input_size, hidden_size)的随机数组，二维表
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)

        #形状为(1, hidden_size)的全零数组
        self.bias_hidden = np.zeros((1, hidden_size))

        #形状为(hidden_size, output_size)的随机数组，二维表
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        #形状为(1, output_size)的全零数组
        self.bias_output = np.zeros((1, output_size))

    # 前向传播
    def forward(self, X):
        """
        前向传播函数

        参数:
        X -- 输入数据

        返回值:
        返回神经网络的预测结果
        """
        # 将输入数组和权重数组相乘，再加偏置
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden

        #代码将神经网络隐藏层的输入（self.hidden_input）通过sigmoid激活函数进行计算，并将结果赋值给self.hidden_output。sigmoid函数常用于神经网络中，它能将连续的实数映射到(0, 1)区间
        self.hidden_output = sigmoid(self.hidden_input)

        # 将隐藏层和权重相乘，再加偏置
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predictions = sigmoid(self.final_input)
        return self.predictions

    def train(self, X, y, learning_rate=0.01, epochs=10000):
        """
        训练模型函数

        参数:
        X -- 输入数据
        y -- 输出数据
        learning_rate -- 学习率
        epochs -- 迭代次数

        返回值:
        无
        """
        # 训练模型
        for epoch in range(epochs):
            # 前向传播
            self.forward(X)

            # 计算损失
            #该函数计算二元交叉熵损失，其中y是真实标签，self.predictions是模型预测的概率。通过计算预测值与真实值之间的差异，来评估模型的性能
            loss = -np.mean(y * np.log(self.predictions) + (1 - y) * np.log(1 - self.predictions))

            # 反向传播
            #这个函数计算了两个变量的差值，并使用矩阵乘法计算了隐藏层误差，其中误差被缩放并根据隐藏层的输出值进行调整。
            output_error = self.predictions - y
            hidden_error = np.dot(output_error, self.weights_hidden_output.T) * (
                        self.hidden_output * (1 - self.hidden_output))

            # 更新权重和偏置
            self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
            self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
            self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
            self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    # 创建样本数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # 创建神经网络模型
    input_size = X.shape[1]
    hidden_size = 4
    output_size = 1
    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    # 训练模型
    model.train(X, y)

    # 进行预测
    predictions = model.forward(X)
    print("Predictions:")
    print(predictions)