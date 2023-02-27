import csv
import numpy
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter


def sigmoid(x):
    x[x < -700] = -700
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, number_features, learning_rate=0.1):
        self.learning_rate = learning_rate
        limit = np.sqrt(1 / number_features)
        self.w = np.random.uniform(-limit, limit, (number_features, 1))
        self.number_features = number_features

    def fit(self, train_data, y, iterations):
        number_train_data = train_data.shape[0]
        y = np.reshape(y, (number_train_data, 1))

        for _ in range(iterations):
            out = train_data.dot(self.w)
            prediction = sigmoid(out)
            dw = train_data.T.dot(prediction - y)
            self.w = self.w - self.learning_rate * dw

    def predict(self, test_data):
        out = test_data.T.dot(self.w)
        # prediction = np.round(sigmoid(out))
        prediction = float(sigmoid(out))
        return prediction


class KNN:
    def __init__(self, train_features, train_labels, k=1):
        self.train_features = train_features
        self.train_labels = train_labels
        self.k = k
        self.len = len(train_labels)

    def predict(self, test_feature):
        knn_points = np.ones([self.k, 2]) * 65535
        for i in range(self.len):
            max_index = np.argmax(knn_points, axis=0)[0]
            dist = np.linalg.norm(test_feature - self.train_features[i])
            if knn_points[max_index][0] > dist:
                knn_points[max_index] = (dist, self.train_labels[i])
        ones_num = np.count_nonzero(knn_points[:, 1] == 1.)
        zeros_num = self.k - ones_num
        return ones_num / self.k
        # if ones_num > zeros_num:
        #     return 1
        # else:
        #     return 0


email_data = []
label = []
with open('data/emails.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    ori_email_header = next(csv_reader)
    for row in csv_reader:
        email_data.append(row[1:3001])
        label.append(int(row[3001]))

email_data = [[float(x) for x in row] for row in email_data]
email_data = np.array(email_data)
label = np.array(label)
# print(email_data.shape)
# print(label.shape)
fold_num = 1000
i = 0
train_slice = [*range(0, i * fold_num)] + [*range(i * fold_num + fold_num, 5 * fold_num)]
test_slice = [*range(i * fold_num, i * fold_num + fold_num)]
# print(train_slice)
# print(test_slice)
logistic_regression_model = LogisticRegression(number_features=3000)
logistic_regression_model.fit(email_data[train_slice], label[train_slice], 5000)
test_points = email_data[test_slice]
test_labels = label[test_slice]

log_score = []
log_label = []
for _ in range(fold_num):
    predict_result = logistic_regression_model.predict(test_points[_])
    true_label = int(label[_])
    log_score.append(predict_result)
    log_label.append(true_label)

knn_model = KNN(email_data[train_slice], label[train_slice], k=5)
test_points = email_data[test_slice]
test_labels = label[test_slice]

knn_score = []
knn_label = []
for _ in range(fold_num):
    predict_result = knn_model.predict(test_points[_])
    true_label = int(label[_])
    knn_score.append(predict_result)
    knn_label.append(true_label)

log_fpr, log_tpr, log_thresholds = roc_curve(log_label, log_score)
log_auc = auc(log_fpr, log_tpr)
knn_fpr, knn_tpr, knn_thresholds = roc_curve(knn_label, knn_score)
knn_auc = auc(knn_fpr, knn_tpr)

log_label = "Logistic Regression AUC="
log_label += "{:.3}".format(log_auc)

knn_label = "KNN AUC="
knn_label += "{:.3}".format(knn_auc)
plt.plot(log_fpr, log_tpr, 's-', color='r', label=log_label)
plt.plot(knn_fpr, knn_tpr, 's-', color='b', label=knn_label)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()


