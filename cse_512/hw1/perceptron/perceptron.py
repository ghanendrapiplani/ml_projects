import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import sys

print(sys.argv)

np.random.seed(10)
fileloc = sys.argv[2]
data = pd.read_csv(fileloc)
data.sort_values(by=[data.columns[len(data.columns) - 1]], ascending=True, inplace=True)
erm_normal = False

if sys.argv[4] == "erm":
    erm_normal = True

folds = 10

columns = list(data.columns)
y_colName = data.columns[len(data.columns) - 1]
columns.remove(y_colName)


def run_algo(X, y):
    w = np.zeros(len(columns) + 1)
    w = w.reshape(len(columns) + 1, 1)
    loop = True
    epochs = 0
    total_times_to_run = 1000

    while loop and epochs < total_times_to_run:
        classifications_done = True
        for i in range(len(X)):
            c_y, c_r = y[i], X[i]
            c_r = np.insert(c_r, 0, 1).reshape(1, len(columns) + 1)
            if ((np.dot(c_r, w)) * c_y) <= 0:
                w += np.dot(c_r, c_y).reshape(len(columns) + 1, 1)
                classifications_done = False
        if classifications_done:
            loop = False
        epochs += 1
    w = w.reshape(len(columns) + 1)
    print("Weights = {}".format(w))
    return w, epochs


def fx_erm(X, y, w):
    len_w = len(columns) + 1
    w = w.reshape(len_w, 1)
    err_count = 0
    for i in range(len(X)):
        curr_y = y[i]
        curr_row = X[i]
        curr_row = np.insert(curr_row, 0, 1)
        curr_row = curr_row.reshape(1, len_w)
        if ((np.dot(curr_row, w)) * curr_y) <= 0:
            err_count += 1

    return err_count / len(X)


def plot_graph(w, X, y):
    x_points = np.linspace(0, 1, 10)
    y_ = -(w[1] * x_points + w[0]) / w[2]
    plt.plot(x_points.T, y_.T, color="red")
    plt.plot(X[:592, 0], X[:592, 1], '+', color='blue', label='0')
    plt.plot(X[592:, 0], X[592:, 1], '+', color='orange', label='1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Perceptron for linear-separable-data")
    plt.legend()
    plt.show()


X = data[columns].to_numpy()
y = data[y_colName].to_numpy()
y = np.array([1 if i == 1 else -1 for i in y])

if erm_normal:
    w, epochs = run_algo(X, y)
    print("Total Epochs = {} ".format(epochs))
    print("ERM for data = {}".format(fx_erm(X, y, w)))
    if "breast" not in fileloc: plot_graph(w, X, y)


def splitData(data, k=10):
    bucket_list = []
    n = int(data.shape[0] / k)
    indexList = list(range(0, data.shape[0]))

    for i in range(k):
        count = 0
        bkt = []
        while count < n:
            index = rd.randint(0, data.shape[0] - 1)
            if index in indexList:
                indexList.remove(index)
                bkt.append(data.loc[index].to_list())
                count += 1
        bucket_list.append(bkt)

    if len(indexList) > 0:
        item = 0
        for index in indexList:
            bucket_list[item].append(data.loc[index].to_list())
            item += 1
            if item == k:
                item = 0

    return bucket_list


def crossVal(buckets, data):
    err_list = []
    epochtotal = 0
    for i in range(len(buckets)):
        print("Round {}".format(i + 1))
        bkts = []
        for target in range(len(buckets)):
            if target != i: bkts += buckets[target]
        finalCols = list(data.columns)
        data_test = pd.DataFrame(bkts, columns=finalCols)
        X = data_test[columns].to_numpy()
        y = data_test[y_colName].to_numpy()
        y = np.array([1 if i == 1 else -1 for i in y])
        w, epochs = run_algo(X, y)
        epochtotal += epochs
        X = data_test[columns].to_numpy()
        y = data_test[y_colName].to_numpy()
        y = np.array([1 if i == 1 else -1 for i in y])
        err = fx_erm(X, y, w)
        err_list.append(err)
        print("ERM = {}\n".format(err))
    print("Total epochs = {}".format(epochtotal))
    return err_list


if not erm_normal:
    buckets = splitData(data, 10)
    print("mean_err={}".format(np.mean(crossVal(buckets, data))))
