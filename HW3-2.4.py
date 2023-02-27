import csv
import numpy as np
import matplotlib.pyplot as plt


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
        if ones_num > zeros_num:
            return 1
        else:
            return 0


email_data = []
label = []
with open('data/emails.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    ori_email_header = next(csv_reader)
    for row in csv_reader:
        email_data.append(row[1:3001])
        label.append(row[3001])

email_data = [[float(x) for x in row] for row in email_data]
email_data = np.array(email_data)
label = np.array(label)
# print(email_data.shape)
# print(label.shape)
fold_num = 1000
average_accuracies = []
ks = [1, 3, 5, 7, 10]
for _ in range(5):
    k = ks[_]
    accuracies = []
    for i in range(5):
        train_slice = [*range(0, i * fold_num)] + [*range(i * fold_num + fold_num, 5 * fold_num)]
        test_slice = [*range(i * fold_num, i * fold_num + fold_num)]
        # print(train_slice)
        # print(test_slice)

        knn_model = KNN(email_data[train_slice], label[train_slice], k=k)
        test_points = email_data[test_slice]
        test_labels = label[test_slice]
        # print(test_points.shape)
        # print(label[1])
        Tp = 0
        Fp = 0
        Tn = 0
        Fn = 0
        for _ in range(fold_num):
            predict_result = int(knn_model.predict(test_points[_]))
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
        accuracies.append(accuracy)
    average_accuracy = sum(accuracies) / 5
    average_accuracies.append(average_accuracy)
    print("When k is", k, ", average accuracy is ", "{:.3}".format(average_accuracy))

plt.plot(ks,average_accuracies, 's-', color='r')
plt.xlabel("k")
plt.ylabel("accuracy")
plt.title('kNN 5-fold Cross validation')
# plt.legend()
plt.show()
