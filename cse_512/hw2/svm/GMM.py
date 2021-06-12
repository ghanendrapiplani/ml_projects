import argparse
import numpy as np
import math
import csv
from sklearn.cluster import KMeans


class GMMClass:
    def __init__(self, X, K=3):
        self.k = int(K)
        self.weights, self.centers, self.covarianceMatrix = [], [], []

        m, numFeatures = X.shape

        for _ in range(self.k):
            self.weights.append(1.0 / self.k)

        covariance = np.cov(X) * (m - 1)

        for _ in range(self.k):
            covarianceMatrix = np.zeros((numFeatures, numFeatures), dtype=float)
            for i in range(numFeatures):
                covarianceMatrix[i][i] = covariance[i][i]
            self.covarianceMatrix.append(covarianceMatrix)

        if self.k == 1:
            self.centers.append(np.mean(X, axis=0))
        else:
            kMean = KMeans(n_clusters=self.k, init='k-means++').fit(X)
            for j in range(self.k):
                self.centers.append(kMean.cluster_centers_[j, :])

    def update(self, componentsWeights, componentsCenters, compCovarMat):
        self.weights = componentsWeights
        self.centers = componentsCenters
        self.covarianceMatrix = compCovarMat


class GMMFullDigits:
    def __init__(self, K=3):
        self.components = int(K)
        self.GMMForDigits = None

    def fit(self, trainingData, Y):
        self.GMMForDigits = []
        maxEpochs = 1000

        for digit in range(10):
            X = getTrainingData(trainingData, Y, digit)
            m, N = X.shape

            singleDigitGMM = GMMClass(X, self.components)
            componentsWeights = singleDigitGMM.weights
            componentsCenters = singleDigitGMM.centers
            compCovarMat = singleDigitGMM.covarianceMatrix

            r = np.zeros((self.components, m), dtype=float)
            for _ in range(maxEpochs):
                for i in range(self.components):
                    invCovarianceMat = np.linalg.pinv(singleDigitGMM.covarianceMatrix[i])
                    determinant = np.linalg.det(singleDigitGMM.covarianceMatrix[i]) + 0.0000001
                    for j in range(m):
                        xCenter = np.subtract(X[j, :], singleDigitGMM.centers[i])
                        xCenter = xCenter.reshape(1, 64)
                        te = np.dot(xCenter, np.dot(invCovarianceMat, np.transpose(xCenter)))

                        r[i][j] = componentsWeights[i] * ((1 / (2 * math.pi)) ** (N / 2)) * (
                                1.0 / math.sqrt(determinant)) * \
                                  math.exp(-0.5 * te[0][0])

                    totalSum = np.sum(r, axis=1)
                    r[i, :] = r[i, :] / totalSum[i]

                for i in range(self.components):
                    sumTemp = 0
                    sumr = 0
                    covTemp = np.zeros((N, N), dtype=float)
                    for j in range(m):
                        sumTemp += r[i][j] * X[j, :]
                        sumr += r[i][j]
                        covTemp = np.add(((np.dot(np.transpose(X[j, :] - componentsCenters[i]),
                                                  X[j, :] - componentsCenters[i])) * r[i][j]), covTemp)
                    componentsCenters[i] = sumTemp / sumr
                    componentsWeights[i] = sumr / m
                    compCovarMat[i] = covTemp / sumr

            singleDigitGMM.update(componentsWeights, componentsCenters, compCovarMat)
            self.GMMForDigits.append(singleDigitGMM)

    def predict(self, X, Y):
        rightPred = 0
        m = X.shape[0]
        correctPredictions, digitCount = [0] * 10, [0] * 10

        for (x, y) in zip(X, Y):
            predictedDigit = 0
            actualDigit = int(y)
            maxProb = 0
            for digit in range(10):
                probMeasure = 0
                singleDigitGMM = self.GMMForDigits[digit]
                for component in range(singleDigitGMM.k):
                    determinant = np.linalg.det(singleDigitGMM.covarianceMatrix[component]) + 0.0000001
                    invCovarianceMat = np.linalg.pinv(singleDigitGMM.covarianceMatrix[component])
                    xCenter = np.subtract(x, singleDigitGMM.centers[component])
                    xCenter = xCenter.reshape(1, 64)
                    probMeasure += (1.0 / math.sqrt(determinant)) * \
                                   math.exp(-0.5 *
                                            np.dot(xCenter, np.dot(invCovarianceMat, np.transpose(xCenter)))[0][0])
                if probMeasure > maxProb:
                    maxProb = probMeasure
                    predictedDigit = digit

            if predictedDigit == actualDigit:
                rightPred += 1
                correctPredictions[predictedDigit] += 1

            digitCount[actualDigit] += 1

        accuracy = float(rightPred) / m
        print("{:.2f} % is the overall accuracy of the GMM".format(accuracy * 100))

        for digit in range(10):
            accuracy = (float(correctPredictions[digit]) / digitCount[digit]) * 100
            print("{:.2f} % accuracy for {} digit".format(accuracy, digit))


def getTrainingData(data, labels, digit):
    reqData = []
    for x, y in zip(data, labels):
        if y == digit:
            reqData.append(x)

    trainingDataForDigit = np.array(reqData)
    return trainingDataForDigit


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_component', required=True)
    parser.add_argument('--trainpath', required=True)
    parser.add_argument('--testpath', required=True)
    return parser


if __name__ == "__main__":
    arguments = vars(parser().parse_args())
    componentsInGMM = arguments['num_component']
    dataPath = arguments['trainpath']
    testDataPath = arguments['testpath']

    with open(dataPath, 'r') as file:
        input = list(csv.reader(file, delimiter=','))

    trainingData = np.array(input, dtype=float)

    X, y = trainingData[:, :-1], trainingData[:, -1]

    gmmForAllDigit = GMMFullDigits(componentsInGMM)
    gmmForAllDigit.fit(X, y)

    with open(testDataPath, 'r') as file:
        inputTest = list(csv.reader(file, delimiter=','))

    testData = np.array(inputTest, dtype=float)
    X, y = testData[:, :-1], testData[:, -1]

    gmmForAllDigit.predict(X, y)
