import numpy as np
import argparse
from csv import reader


def probDistFx(x, mean, covar):
    x_mu = (x - mean)
    c = (1 / ((np.linalg.det(covar) + 0.0000001) ** (1 / 2)) * ((np.pi * 2) ** (covar.shape[0] / 2)))
    return c * np.exp((-1 / 2) * np.sqrt(np.sum(np.square(
        np.matmul(np.matmul(x_mu, np.linalg.pinv(covar)), x_mu.T)), axis=-1)))


class GMMClass:
    def __init__(self, k=1):
        self.k = k
        self.w = np.ones(k) / k
        self.mean, self.covariance = None, None

    def fit(self, X):
        X = np.array(X)
        self.mean = np.random.choice(X.flatten(), (self.k, X.shape[1]))

        covariance = []
        for i in range(self.k):
            covariance.append(np.cov(X, rowvar=False))
        covariance = np.array(covariance)
        max_epochs = 100
        for step in range(max_epochs):
            e = 1e-8
            print("Epoch {}".format(step))
            likelihoodList = []
            for j in range(self.k):
                likelihoodList.append(probDistFx(X, self.mean[j], covariance[j]) + e)
            likelihoodList = np.array(likelihoodList)

            for j in range(self.k):
                denominator = np.sum([likelihoodList[i] * self.w[i] for i in range(self.k)], axis=0) + e
                r = ((likelihoodList[j] * self.w[j]) / denominator)
                r_ = r.reshape(len(X), 1)
                x_ = X - self.mean[j]
                self.mean[j] = np.sum(r_ * X, axis=0) / (np.sum(r + e))
                covariance[j] = np.dot((r_ * x_).T, x_) / (np.sum(r) + e)
                self.w[j] = np.mean(r)

        self.covariance = covariance

    def predictFx(self, X):
        X = np.array(X)
        pred = 0
        for j in range(self.k):
            pred = pred + self.w[j] * probDistFx(x=X, mean=self.mean[j], covar=self.covariance[j])
        return pred


def readCsv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for data_row in csv_reader:
            data.append(data_row)
    return data


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    return parser


if __name__ == "__main__":
    arguments = vars(parser().parse_args())

    train = readCsv(arguments['train'])
    component = int(arguments['component'])

    trainDataMap = {}
    for i in range(10):
        trainDataMap[i] = []

    for row in train:
        y = int(row[64])
        trainDataMap[y].append([float(x.strip()) for x in row[:-1]])

    gmmList = []
    for i in range(10):
        gmm = GMMClass(component)
        gmm.fit(trainDataMap[i])
        gmmList.append(gmm)

    test = readCsv(arguments['test'])

    predictions = {}
    for i in range(10):
        predictions[i] = []

    for row in test:
        y_actual = int(row[64])
        maxProb = float('-inf')
        y_predicted = -1
        for index in range(len(gmmList)):
            print("Predicting for {}".format(index))
            p = gmmList[index].predictFx([float(x.strip()) for x in row[:-1]])
            if np.sum(p) > maxProb:
                y_predicted = index
                maxProb = np.sum(p)
        accuracy = 0
        if y_actual == y_predicted:
            accuracy = 1
        predictions[y_actual].append(accuracy)

    total = 0
    print("\n========================= Result =========================")
    for index in range(len(predictions)):
        s = np.sum(np.array(predictions[index]))
        print("{:.2f} % accuracy for {} digit".format((s * 100) / len(predictions[index]), index))
        total += s
    print("{:.2f} % is the overall accuracy of the GMM".format(total * 100 / len(test)))
