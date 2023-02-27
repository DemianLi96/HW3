import numpy as np
import matplotlib.pyplot as plt

data = []
with open('data/D2z.txt') as f:
    for line in f:
        words = line.split()
        data.append([float(words[0]), float(words[1]), int(words[2])])
data = np.array(data)


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
            return 1.
        else:
            return 0.


knn_model = KNN(data[:, 0:2], data[:, 2], k=1)
test_point = np.array([0.596728, 0.112458])
test_data = []
for _ in range(50):
    test1 = -2 + np.random.sample() * 4
    test2 = -1.5 + np.random.sample() * 3
    test_point = [test1, test2]
    result = knn_model.predict(test_point)
    test_data.append([test1, test2, result])
test_data = np.array(test_data)

zero_train_slice = data[:, 2] == 0
one_train_slice = data[:, 2] == 1
zero_test_slice = test_data[:, 2] == 0
one_test_slice = test_data[:, 2] == 1

plt.scatter(data[zero_train_slice][:, 0], data[zero_train_slice][:, 1], s=80, marker='^',
            label='train data with 0 label', c='r', alpha=1)
plt.scatter(data[one_train_slice][:, 0], data[one_train_slice][:, 1], s=80, marker='o', label='train data with 1 label',
            c='y', alpha=1)
plt.scatter(test_data[zero_test_slice][:, 0], test_data[zero_test_slice][:, 1], s=50, marker='^',
            label='test data with 0 label', c='b', alpha=0.5)
plt.scatter(test_data[one_test_slice][:, 0], test_data[one_test_slice][:, 1], s=50, marker='o',
            label='test data with 1 label', c='k', alpha=0.5)
plt.legend()
plt.show()
