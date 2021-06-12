import numpy as np
import math
import matplotlib.pyplot as plt
from csv import reader
import sys
from random import randrange
from random import seed


def cross_validation_split(data, fold_count):
    data_split = []
    data_copy = list(data)
    split_size = int(len(data) / fold_count)
    for i in range(fold_count):
        split = list()
        while len(split) < split_size:
            index = randrange(len(data_copy))
            split.append(data_copy.pop(index))
        data_split.append(split)
    return data_split


def gettraintestdata(X, y, test_size=0.2, seed=None):
    np.random.seed(seed)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    split_index = len(y) - int(len(y) // (1 / test_size))
    train_x, test_x = X[:split_index], X[split_index:]
    train_y, test_y = y[:split_index], y[split_index:]
    return train_x, test_x, train_y, test_y


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.x_feature_index = None
        self.threshold_val = None
        self.alpha = None


class ADABoost:
    def __init__(self, n_classifier=5):
        self.n_classifier = n_classifier

    def fit(self, X, y):
        n, f_count = np.shape(X)

        w = np.full(n, (1 / n))
        hypothesis = []
        self.classifiers = []
        for _ in range(self.n_classifier):
            classifier = DecisionStump()
            min_error = float('inf')
            for feature in range(f_count):
                f_val = np.expand_dims(X[:, feature], axis=1)
                unique_values = np.unique(f_val)
                for threshold_val in unique_values:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[X[:, feature] < threshold_val] = -1
                    error = sum(w[y != prediction])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        classifier.polarity = p
                        classifier.threshold_val = threshold_val
                        classifier.x_feature_index = feature
                        min_error = error
            classifier.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            predictions = np.ones(np.shape(y))
            indx = (classifier.polarity * X[:, classifier.x_feature_index] < classifier.polarity * classifier.threshold_val)
            predictions[indx] = -1
            w = w*np.exp(-classifier.alpha * y * predictions)
            w = w/np.sum(w)

            self.classifiers.append(classifier)
            hypothesis.append(classifier.alpha)

        return hypothesis

    def predict(self, x_val):
        n = np.shape(x_val)[0]
        y_predicted = np.zeros((n, 1))
        for classifier in self.classifiers:
            predictions = np.ones(np.shape(y_predicted))
            indx = (classifier.polarity * x_val[:, classifier.x_feature_index] < classifier.polarity * classifier.threshold_val)
            predictions[indx] = -1
            y_predicted += classifier.alpha * predictions
        y_predicted = np.sign(y_predicted).flatten()

        return y_predicted


def fx_erm(y_predicted, y_actual):
    predictions = []
    for i in range(len(y_predicted)):
        prediction = 0.0 if y_predicted[i] == y_actual[i] else 1.0
        predictions.append(prediction)
    return sum(predictions) / len(predictions)


def get_data(data):
    x = []
    y = []
    for row in data:
        x.append([p for p in row[:-1]])
        y.append(row[-1])
    return x, y


def perform_adaboost(train_x, test_x, train_y, test_y, tVal):
    classifier = ADABoost(n_classifier=tVal)
    hypothesis = classifier.fit(train_x, train_y)
    y_predicted = classifier.predict(test_x)
    error = fx_erm(y_predicted, test_y)
    return hypothesis, error


def crossVal(data, folds, tVal):
    splits = cross_validation_split(data, folds)
    erm_values = []
    i = 0
    for split in splits:
        train_set = list(splits)
        train_set.remove(split)
        train_set = sum(train_set, [])
        test_set = []
        for row in split:
            row_copy = list(row)
            test_set.append(row_copy)
        train_x, train_y = get_data(train_set)
        test_x, test_y = get_data(test_set)
        train_x = np.array(train_x)
        test_x = np.array(test_x)
        train_y = np.array([1 if p == 1 else -1 for p in train_y])
        test_y = np.array([1 if p == 1 else -1 for p in test_y])

        hypothesis, erm_value = perform_adaboost(train_x, test_x, train_y, test_y, tVal)
        erm_values.append(erm_value)
        print("Fold Number : {}" .format(i+1))
        print("Hypothesis : {}".format(hypothesis))
        print("ERM : {}".format(erm_value))
        print("\n")
        i += 1
    return erm_values


data_loc = sys.argv[2]
erm_normal = False
kfold = False
analysis = False

if sys.argv[4] == "erm":
    erm_normal = True
if sys.argv[4] == "kfold":
    kfold = True
if sys.argv[4] == "analysis":
    analysis = True

seed(1)


def load_csv(inp_file):
    data = []
    with open(inp_file, 'r') as file:
        csv = reader(file)
        for row in csv:
            data.append(row)
    return data


data = load_csv(data_loc)

data[0] = [x.strip() for x in data[0]]
i = 1
for row in data[1:]:
    data[i] = [float(x.strip()) for x in row]
    i = i + 1
data = data[1:]

folds = 10

tVal = 5


def graph(plot_list, xlabel, ylabel, title, fig_name):
    plt.figure(fig_name)
    for plot in plot_list:
        plt.plot(plot[0], plot[1], label=plot[2])
    x_axis = list(set(plot_list[0][0][:-1]))
    y_axis = list(set(plot_list[0][1]))
    plt.xticks(x_axis, rotation=90)
    plt.yticks(y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=2)
    plt.show()


if erm_normal:
    X, y = get_data(data)
    x_ = np.array(X)
    y_ = np.array([1 if p == 1 else -1 for p in y])
    train_x, test_x, train_y, test_y = gettraintestdata(x_, y_, test_size=0.3)
    hypothesis, erm_value = perform_adaboost(train_x, test_x, train_y, test_y, tVal)
    print("ERM mode")
    print("t_val = {}".format(tVal))
    print("Hypothesis: {}".format(hypothesis))
    print("ERM: {}".format(erm_value))
elif kfold:
    erm = crossVal(data, folds, tVal)
    print("Cross validation mode for {} folds".format(folds))
    print("t_val = {}".format(tVal))
    print("Error values: {}".format(erm))
    print("Average error: {}" .format(sum(erm) / folds))
elif analysis:
    print("Comparative analysis:")
    X, y = get_data(data)
    x = np.array(X)
    y = np.array([1 if p == 1 else -1 for p in y])
    erm_vals = []
    cross_vals = []
    T = 10
    train_x, test_x, train_y, test_y = gettraintestdata(x, y, test_size=0.3)
    for i in range(T):
        t_val = i
        print("T value: {}".format(t_val))
        erm_vals.append(perform_adaboost(train_x, test_x, train_y, test_y, t_val)[1])
        cross_vals.append((sum(crossVal(data, folds, t_val)) / int(folds)))

    x_val = list(range(1, T + 1))
    plots = [[x_val, erm_vals, "ERM over entire data"],
             [x_val, cross_vals, "Cross Validation ERM {} fold".format(folds)]]
    graph(plots, 'T value', 'ERM', 'Q2 Graph', 'Comparative Analysis of ERM over entire data vs Cross validation ERM')


