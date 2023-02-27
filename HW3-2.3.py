import csv
import numpy as np


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
        prediction = np.round(sigmoid(out))
        return prediction


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
for i in range(5):
    train_slice = [*range(0, i * fold_num)] + [*range(i * fold_num + fold_num, 5 * fold_num)]
    test_slice = [*range(i * fold_num, i * fold_num + fold_num)]
    # print(train_slice)
    # print(test_slice)
    logistic_regression_model = LogisticRegression(number_features=3000)
    logistic_regression_model.fit(email_data[train_slice], label[train_slice], 5000)
    test_points = email_data[test_slice]
    test_labels = label[test_slice]
    # print(test_points.shape)
    # print(label[1])
    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0
    for _ in range(fold_num):
        predict_result = int(logistic_regression_model.predict(test_points[_]))
        true_label = int(label[_])
        # print(_)
        if predict_result == 1 and true_label == 1:
            Tp += 1
        elif predict_result == 1 and true_label == 0:
            Fp += 1
        elif predict_result == 0 and true_label == 1:
            Fn += 1
        elif predict_result == 0 and true_label == 0:
            Tn += 1
    accuracy = (Tp + Tn) / (Tp + Tn + Fp + Fn)
    precision = Tp / (Tp + Fp + 1e-10)
    recall = Tp / (Tp + Fn)
    print("Fold ", i + 1, "'s accuracy", "{:.3}".format(accuracy), ", precision is ", "{:.3}".format(precision),
          ", recall is ", "{:.3}".format(recall))
