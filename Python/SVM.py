import numpy as np
from DataPreparation import SetHolder


class AbstractSvm:
    def __init__(self, y, cl):
        self.kernel_matrix = None
        self.obs_kernel = None
        self.alpha = None
        self.b = 0.0
        self.y = np.ones(len(y), dtype=int)
        self.y[y != cl] = -1
        self.cl = cl

    @staticmethod
    def calc_kernel_matrix(x, z, sigma):
        f = np.vectorize(lambda i, j: np.exp(-(np.linalg.norm(x[i] - z[j]) ** 2) / (2 * sigma ** 2)))
        return np.fromfunction(f, shape=(len(x), len(z)), dtype=int)

    def descent(self, c, rate, iter_nr):
        y_j_y_k = np.outer(self.y, self.y).astype(float) * self.kernel_matrix
        ones = np.ones(len(self.y))
        self.alpha = np.random.random(len(self.y))
        for _ in range(iter_nr):
            gradient = ones - np.dot(y_j_y_k, self.alpha)
            self.alpha += rate * gradient
            self.alpha[self.alpha > c] = c
            self.alpha[self.alpha < 0.0] = 0.0
        self.b = np.mean(self.y - np.dot(self.alpha * self.y, self.kernel_matrix))

    def predict(self, full=False):
        if not full:
            idx = np.argwhere(self.y == 1).flatten()
            res = (self.alpha[idx] @ self.obs_kernel[idx] + self.b) / (self.alpha @ self.obs_kernel)
            # print(res)
            return res
        else:
            return np.sign((self.alpha * self.y) @ self.obs_kernel + self.b)


# noinspection PyUnresolvedReferences
class MultiClassSvm:
    def __init__(self, x, y, strategy, sigma=1.0, c=1.0):
        if strategy not in ['ovo', 'ova']:
            raise Exception
        self.strategy = strategy
        self.x = x
        class_nr = len(np.unique(y))
        if strategy == 'ovo':
            self.y = y
            self.class_nr = class_nr
        self.svm_list = []
        kernel = AbstractSvm.calc_kernel_matrix(x, x, sigma)
<<<<<<< HEAD
        if strategy == 'ova':
            for i in range(class_nr):
                self.svm_list.append(AbstractSvm(y, i))
                self.svm_list[-1].kernel_matrix = kernel.view()
        else:
            for i in range(class_nr):
                for j in range(i):
                    idx = np.hstack([np.argwhere(y == i).flatten(), np.argwhere(y == j).flatten()])
                    self.svm_list.append(AbstractSvm(y[idx], j))
                    self.svm_list[-1].kernel_matrix = kernel[np.ix_(idx, idx)]
=======
        for svm in self.svm_list:
            svm.kernel_matrix = kernel.view()
>>>>>>> 1da9fd8f5636994a8b71f1a06c0276a9b4750ec7
        self.sigma = sigma
        self.c = c

    def fit(self, rate=0.001, iter_nr=1000):
        for svm in self.svm_list:
            svm.descent(self.c, rate, iter_nr)

    def predict(self, x):
        obs_kernel = AbstractSvm.calc_kernel_matrix(self.x, x, self.sigma)
<<<<<<< HEAD
        if self.strategy == 'ova':
            predictions = np.ndarray(shape=(len(self.svm_list), len(x)))
            for index, svm in enumerate(self.svm_list):
                svm.obs_kernel = obs_kernel.view()
                predictions[index] = svm.predict()
            return np.argmax(predictions, axis=0)
        else:
            counters = np.zeros(shape=(len(x), self.class_nr), dtype=int)
            index = 0
            for i in range(self.class_nr):
                for j in range(i):
                    self.svm_list[index].obs_kernel = obs_kernel[np.hstack([np.argwhere(self.y == i).flatten(),
                                                                            np.argwhere(self.y == j).flatten()])]
                    predictions = self.svm_list[index].predict(full=True)
                    for k, p in enumerate(predictions):
                        if p == 1:
                            counters[k][j] += 1
                        else:
                            counters[k][i] += 1
                    index += 1
            # print(counters)
            return np.argmax(counters, axis=1)
=======
        predictions = np.ndarray(shape=(len(self.svm_list), len(x)))
        for index, svm in enumerate(self.svm_list):
            svm.obs_kernel = obs_kernel.view()
            predictions[index] = svm.predict()
        return np.argmax(predictions, axis=0)
>>>>>>> 1da9fd8f5636994a8b71f1a06c0276a9b4750ec7

    def check(self, x, y):
        return np.sum((self.predict(x) == y).astype(int)) * 100 / len(y)


if __name__ == '__main__':
    data = SetHolder()
    data.normalize()
    svm = MultiClassSvm(data.train_x, data.train_y, 'ovo')
    svm.fit()
    print(svm.check(data.test_x, data.test_y))
<<<<<<< HEAD
=======
    
>>>>>>> 1da9fd8f5636994a8b71f1a06c0276a9b4750ec7
