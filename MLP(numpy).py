"""
    这是一个用神经网络进行红酒品质评价的程序
    所用的神经网络非常简单，只有一层隐藏层（含8个神经元）
    最终测试的结果：
        共1599条数据，取前1400条数据作训练集。500轮迭代，在训练集上准确率为89%左右，验证集上准确率为86%
        不过这是严格要求后的结果，如果是四舍五入（epsilon = 0.5）可以达到100%的准确率
        完整的结果和训练后的参数在output.txt中
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ---神经元
def sigmoid(inputs):
    """f(x) = 1 / (1+e^(-x))"""
    return 1 / (1 + np.exp(-inputs))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


def accuracy(all_y_trues, y_preds):
    # 确保输入长度一致
    if len(all_y_trues) != len(y_preds):
        raise ValueError("输入的标签数量不一致")

    # 计算正确预测的数量
    epsilon = 0.1
    correct_predictions = sum(1 for true, pred in zip(all_y_trues, y_preds) if abs(true - pred) <= epsilon)

    # 计算准确率
    accuracy_rate = correct_predictions / len(all_y_trues) * 100

    return accuracy_rate


class Neuron:
    def __init__(self, dimension=11):
        self.dimension = dimension
        self.sum = 0
        # 随机初始化
        self.weights = np.random.normal(0, 1, size=self.dimension)
        self.bias = np.random.normal()

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        self.sum = total
        return sigmoid(total)


# ---自定义神经网络
class NeuralNetwork:
    def __init__(self):
        self.output = Neuron(8)  # 输出层
        self.neuron_list = []  # 隐藏层
        for i in range(8):
            self.neuron_list.append(Neuron(11))

    def feedforward(self, inputs):
        hidden_layer_output = []
        # print("inputs=",inputs)
        for neuron in self.neuron_list:
            hidden_layer_output.append(neuron.feedforward(inputs))

        hidden_layer_output = np.array(hidden_layer_output)
        return self.output.feedforward(hidden_layer_output)

    def train(self, train_data, all_y_trues, save_result):
        """
            - train_data is a (n x 11) numpy array, n = # of samples in the dataset.
            - all_y_trues is a numpy array with n elements.
              Elements in all_y_trues correspond to those in train_data.
        """
        derivatives_dict = {}
        learn_rate = 0.01
        epochs = 500  # number of times to loop through the entire dataset
        with open("output.txt", "a") as file:
            for epoch in range(epochs):
                for x, y_true in zip(train_data, all_y_trues):
                    # --- Do a feedforward (we'll need these values later)
                    y_pred = self.feedforward(x)

                    # --- Calculate partial derivatives.
                    # --- Naming: dL_dw1 represents "partial L / partial w1"
                    derivatives_dict['dL_dypred'] = -2 * (y_true - y_pred)

                    # Neuron output
                    dersig_output = deriv_sigmoid(self.output.sum)
                    for i in range(self.output.dimension):  # 权重的偏导
                        derivatives_dict['dypred_dw' + str(11 * 8 + i)] = (self.neuron_list[i].feedforward(
                            x)) * dersig_output
                        derivatives_dict['dypred_dh' + str(i)] = self.output.weights[i] * dersig_output
                    derivatives_dict['dypred_db8'] = dersig_output  # b的偏导
                    # Hidden Layer Neuron
                    for i in range(self.output.dimension):
                        dersig_sumi = deriv_sigmoid(self.neuron_list[i].sum)
                        for j in range(11):
                            derivatives_dict['dh' + str(i) + '_dw' + str(j)] = x[j] * dersig_sumi
                        derivatives_dict['dh' + str(i) + '_db' + str(i)] = dersig_sumi

                    # --- Update weights and biases
                    ratio = learn_rate * derivatives_dict['dL_dypred']
                    # Hidden Layer
                    for i in range(8):
                        for j in range(11):
                            self.neuron_list[i].weights[j] -= ratio * derivatives_dict['dypred_dh' + str(i)] * \
                                                              derivatives_dict['dh' + str(i) + '_dw' + str(j)]
                        self.neuron_list[i].bias -= ratio * derivatives_dict['dypred_dh' + str(i)] * derivatives_dict[
                            'dh' + str(i) + '_db' + str(i)]
                    # Output Layer
                    for i in range(8):
                        self.output.weights[i] -= ratio * derivatives_dict['dypred_dw' + str(11 * 8 + i)]
                    self.output.bias -= ratio * derivatives_dict['dypred_db8']

                    # --- Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, train_data)
                    loss = mse_loss(all_y_trues, y_preds)
                    accuracy_rating = accuracy(all_y_trues, y_preds)
                    print(f"Epoch {epoch} loss: {100 * loss:.5f} accuracy_rating: {accuracy_rating:.2f}%")

                    if save_result:
                        file.write(f"Epoch {epoch} loss: {loss} accuracy_rating: {accuracy_rating:.2f}%\n")

            if save_result:
                index = 1
                for neuron in self.neuron_list:
                    file.write(f"Hidden Layer Neuron{index}:\n")
                    file.write(f"weights: {neuron.weights};  bias: {neuron.bias}\n")
                    index += 1
                file.write(f"Output Layer Neuron:\n")
                file.write(f"weights: {self.output.weights};  bias: {self.output.bias}")

    def evaluate(self, evaluate_data, all_y_trues, save_result):
        if len(all_y_trues) != len(evaluate_data):
            raise ValueError("输入的标签数量不一致")

        y_preds = np.apply_along_axis(self.feedforward, 1, evaluate_data)
        loss = mse_loss(all_y_trues, y_preds)
        accuracy_rating = accuracy(all_y_trues, y_preds)
        print(f"\nevaluate:\nloss: {loss} accuracy_rating: {accuracy_rating:.2f}%")
        if save_result:
            with open("output.txt", "a") as file:
                file.write(f"\n\nevaluate:\nloss: {loss} accuracy_rating: {accuracy_rating:.2f}%")


if __name__ == "__main__":
    # ---读取csv文件
    data = pd.read_csv("winequality-red.csv", sep=";")
    wine_data = data.iloc[:, :-1].values
    quality = data.iloc[:, -1].values
    quality = quality * 0.1

    # ---数据归一化
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    # 对 wine_data 进行归一化
    wine_data_normalized = scaler.fit_transform(wine_data)

    wine_train_data = wine_data_normalized[:1400]
    train_data_quality = quality[:1400]

    wine_evaluate_data = wine_data_normalized[1400:]
    evaluate_data_quality = quality[1400:]

    network = NeuralNetwork()
    save = False
    network.train(wine_train_data, train_data_quality, save)
    network.evaluate(wine_evaluate_data, evaluate_data_quality, save)
