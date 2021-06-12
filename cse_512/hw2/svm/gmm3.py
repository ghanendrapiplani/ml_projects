import numpy as np
import argparse
import pandas as pd


def prob_density_func(x_col, centroid, cov_mat):
    denominator = (1 / ((np.linalg.det(cov_mat) + 0.0000001) ** 0.5) * ((np.pi * 2) ** (cov_mat.shape[0] / 2)))
    x_dist = (x_col - centroid)
    pdf = denominator * np.exp((-1 / 2) * (np.sqrt(np.sum(np.square(np.matmul(np.matmul(x_dist, np.linalg.pinv(cov_mat)), x_dist.T)), axis=-1))))
    return pdf


class GMM:
    def __init__(self, k=3):
        self.k = k
        self.comps_weight = np.ones((k)) / k
        self.comps_centroid = None
        self.comps_cov_mat = None

    def fit(self, x_col):
        x_col = np.array(x_col)
        self.comps_centroid = np.random.choice(x_col.flatten(), (self.k, x_col.shape[1]))

        cov_list = []
        for i in range(self.k):
            cov_list.append(np.cov(x_col, rowvar=False))
        cov_list = np.array(cov_list)
        assert cov_list.shape == (self.k, x_col.shape[1], x_col.shape[1])
        eps = 1e-8
        max_epochs = 10
        for epochs in range(max_epochs):
            mle = []
            for j in range(self.k):
                mle.append(prob_density_func(x_col=x_col, centroid=self.comps_centroid[j], cov_mat=cov_list[j]) + eps)
            mle = np.array(mle)
            assert mle.shape == (self.k, len(x_col))

            for j in range(self.k):
                b = ((mle[j] * self.comps_weight[j]) / (
                        np.sum([mle[i] * self.comps_weight[i] for i in range(self.k)], axis=0) + eps))
                self.comps_centroid[j] = np.sum(b.reshape(len(x_col), 1) * x_col, axis=0) / (np.sum(b + eps))
                cov_list[j] = np.dot((b.reshape(len(x_col), 1) * (x_col - self.comps_centroid[j])).T, (x_col - self.comps_centroid[j])) / (
                        np.sum(b) + eps)
                self.comps_weight[j] = np.mean(b)

        self.comps_cov_mat = cov_list

    def predict(self, x_test):
        x_test = np.array(x_test)
        pdf = 0
        for j in range(self.k):
            pdf += self.comps_weight[j] * prob_density_func(x_col=x_test, centroid=self.comps_centroid[j], cov_mat=self.comps_cov_mat[j])
        return pdf

# def csv_loader(file_path):
#     data = []
#     with open(file_path, 'r') as file:
#         rowwise_data = csv.reader(file)
#         for data_row in rowwise_data:
#             data.append(data_row)
#     return data

parser = argparse.ArgumentParser()
parser.add_argument('--components', required=True)
parser.add_argument('--train', required=True)
parser.add_argument('--test', required=True)
args = vars(parser.parse_args())
comps = int(args['components'])
# train_dataset = csv_loader(args['train'])
# test_dataset = csv_loader(args['test'])
dataset = pd.read_csv(args['train'])
train_dataset = np.array(dataset, dtype=float)
x_col = train_dataset[:, :-1]
y_col = train_dataset[:, -1]
test_dataset = pd.read_csv(args['test'])
test_data = np.array(test_dataset, dtype=float)
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

# digits_dict = {
#     0: [],
#     1: [],
#     2: [],
#     3: [],
#     4: [],
#     5: [],
#     6: [],
#     7: [],
#     8: [],
#     9: [],
# }

# for row_id in train_dataset:
#     y_col = int(row_id[64])
#     digits_dict[y_col].append([float(dat.strip()) for dat in row_id[:-1]])

list_of_gmm_instances_digitwise = []
for i in range(10):
    gmm = GMM(comps)
    list_of_gmm_instances_digitwise.append(gmm)
    x_train = []
    for j in range(len(x_col)):
        if i == y_col[j]:
            x_train.append(x_col[1])
    gmm.fit(x_train)



pred_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

for row_id in range(len(test_data)):
    y_label = y_test[row_id]
    max_prob = float('-inf')
    predicted = -1
    for dig in range(len(list_of_gmm_instances_digitwise)):
        prob = list_of_gmm_instances_digitwise[dig].predict(x_test[row_id])
        if np.sum(prob) > max_prob:
            predicted = dig
            max_prob = np.sum(prob)
    y_pred = 0
    if y_label == predicted:
        y_pred = 1
    pred_dict[y_label].append(y_pred)

total_count = 0
for dig in range(len(pred_dict)):
    pred = np.sum(np.array(pred_dict[dig]))
    print(dig, ' accuracy-', (pred * 100) / len(pred_dict[dig]))
    total_count += pred

print('mean accuracy', (total_count / len(test_dataset)) * 100)
