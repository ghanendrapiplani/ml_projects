import argparse
import numpy as np
import pandas as pd
import math
import random
import struct


def loadImgData(path):
    path = "data/t10k-images.idx3-ubyte" if path == "test" else "data/train-images.idx3-ubyte"
    with open(path, 'rb') as f:
        magic, count = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        images = np.fromfile(f, dtype= np.dtype(np.uint8).newbyteorder('>'))
        images = images.reshape((count, rows, cols))
    return images


def loadLblData(path):
    path = "data/t10k-labels.idx1-ubyte" if path == "test" else "data/train-labels.idx1-ubyte"
    with open(path, 'rb') as f:
        labels = np.fromfile(f, dtype= np.dtype(np.uint8).newbyteorder('>'))
    return labels


class SvmRBFClass():
    def __init__(self, T=1000, lmbda=0.05, type="linear"):
        self.T = T
        self.type = type
        self.lmbda = lmbda
        self.gamma = 0.8
        self.alpha, self.xTrain, self.yTrain = None, None, None

    def kernel(self, X1, X2):
        return math.exp(-self.gamma * np.linalg.norm(X1 - X2) ** 2)

    def fit(self, X, y):
        values, columns = X.shape
        alpha, beta = np.ones(values), np.ones(values)
        for t in range(1, self.T):
            alpha = beta / (self.lmbda * t)
            index = random.randint(0, values - 1)
            row = X[index]
            score = 0
            for j in range(values):
                if j != index:
                    score += alpha[index] * self.kernel(row, X[j])
            if y[index] * score < 1:
                beta += y[index]
        self.alpha = alpha
        self.xTrain = X
        self.yTrain = y

    def predict(self, X, y):
        values, columns = X.shape
        correct_pred = 0
        for i in range(values):
            score = 0
            for alpha, x_train, yTrain in zip(self.alpha, self.xTrain, self.yTrain):
                score += alpha * yTrain * self.kernel(X[i], x_train)
            if score > 0:
                score = 1
            elif score <= 0:
                score = -1
            if y[i] == score:
                correct_pred += 1
        print(" Correct : ", correct_pred, " Accuracy : ", correct_pred * 100 / values)


def getData(filepathTrain, filepathTest, dataset):
    X = None
    y = None
    if dataset == 'BCD':
        data = pd.read_csv(filepathTrain)
        columns = list(data.columns)
        y_colName = data.columns[len(data.columns) - 1]
        columns.remove(y_colName)
        X = data[columns].to_numpy()
        y = data[y_colName].to_numpy()
        y = np.array([1 if i == 1 else -1 for i in y])
    else:
        if not filepathTest:
            X = loadImgData("train").reshape(-1, 28 * 28)
            y = loadImgData("test").reshape(-1, 28 * 28)
        else:
            X = loadLblData("train")
            y = loadLblData("test")

    x = np.c_[np.ones((X.shape[0])), X]
    pos = []
    neg = []
    for i, label in enumerate(y):
        if np.any(label) == -1:
            neg.append(x[i])
        else:
            pos.append(x[i])

    trdata = []
    for i in range(len(neg)):
      trdata.append(list(neg[i]) + [-1])
    for i in range(len(pos)):
      trdata.append(list(pos[i]) + [1])

    train_data = [np.array([x[:-1] for x in trdata]), [x[-1] for x in trdata]]
    return train_data


def costFxGradient(weights, lambd, x_val, y_val):
    distance = 1 - (y_val * np.dot(x_val, weights))
    dw = np.zeros(len(weights))
    if distance > 0:
        dw[1:] = (2 * lambd * weights[1:]) - (x_val[1:] * y_val)
        dw[0] = - lambd * y_val
    else:
        dw[1:] = 2 * lambd * weights[1:]
    return dw


def train_svm_linear(xTrain, yTrain, max_epochs=1000):
    lmbda = 0.5
    weight_list = []
    weights = np.zeros(xTrain.shape[1])
    for epoch in range(1, max_epochs):
        learnRate = 1 / (lmbda * epoch)

    for i in range(len(xTrain)):
        grad_value = costFxGradient(weights, lmbda, xTrain[i], yTrain[i])
        weights = weights - (learnRate * grad_value)
    weight_list.append(list(weights))

    weights = sum(np.array(weight_list)) / max_epochs

    return weights


def test(xTest, yTest, weights):
    rightPred = 0
    for i in range(len(xTest)):
        prod = sum(np.array(xTest[i]) * weights)
        pred = 1 if prod >= 0 else -1
        if pred == yTest[i]:
            rightPred += 1
    print("Test points total : {}" .format(len(xTest)))
    print("Wrongly classified points: {}" .format(len(xTest) - rightPred))
    return float(rightPred) / float(len(xTest))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', help='Kernel Type: Linear or RBF', required=True)
    parser.add_argument('--train', help='Training data path', required=True)
    parser.add_argument('--test', help='Test data path', required=False)
    parser.add_argument('--dataset', help='BCD or MNIST', required=True)
    parser.add_argument('--output', help='Output weight file path', required=True)
    return vars(parser.parse_args())


def getRBFdata(filepath):
    data = pd.read_csv(filepath)
    columns = list(data.columns)
    y_colName = data.columns[len(data.columns) - 1]
    columns.remove(y_colName)
    X = data[columns].to_numpy()
    y = data[y_colName].to_numpy()
    y = np.array([1 if i == 1 else -1 for i in y])
    return X, y, columns, y_colName


if __name__ == "__main__":
    args = parse_args()

    dataset = args['dataset']
    kernel = args['kernel']

    if dataset == "BCD":
        if kernel == "linear":
            trdata = getData(args['train'], args['test'], args['dataset'])
            weights = train_svm_linear(trdata[0], trdata[1], max_epochs=10000)
            test_accuracy = test(trdata[0], trdata[1], weights)
            print("Accuracy achieved: {}" .format(test_accuracy * 100))
            print("Weights: ", weights)
            file_name = args['output']
            f = open(file_name, 'w')  # open file in append mode
            f.write(str(weights))
            f.close()
        else:
            xTrain, yTrain, xCols, yCol = getRBFdata(args['train'])
            svm = SvmRBFClass(type=args['kernel'])
            svm.fit(xTrain, yTrain)
            svm.predict(xTrain, yTrain)

    if dataset == "MNIST":
        trainX, testX = getData(args['train'], False, args['kernel'])
        weights = train_svm_linear(trainX, testX)
        rightPred = test(trainX, testX, weights)
        test_accuracy = float(rightPred) / float(len(trainX))

        print("Accuracy achieved: {}%".format(test_accuracy * 100))
        print("Training Error: {}%".format(100 - (test_accuracy * 100)))
        file_name = args['output']
        f = open("train_" + file_name, 'w')
        f.write(str(weights))
        f.close()

        trainX, testX = getData(args['train'], True, args['kernel'])
        weights = train_svm_linear(trainX, testX)
        rightPred = test(trainX, testX, weights)
        test_accuracy = float(rightPred) / float(len(trainX))

        print("Accuracy achieved: {}%".format(test_accuracy * 100))
        print("Training Error: {}%".format(100 - (test_accuracy * 100)))
        file_name = args['output']
        f = open("test_" + file_name, 'w')
        f.write(str(weights))
        f.close()