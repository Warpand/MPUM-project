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

    def predict(self):
        idx = np.argwhere(self.y == 1).flatten()
        predictions = self.alpha[idx] @ self.obs_kernel[idx] + self.b
        return predictions


class MultiClassSvm:
    def __init__(self, x, y, sigma=1.0, c=1.0):
        self.x = x
        class_nr = len(np.unique(y))
        self.svm_list = []
        for i in range(class_nr):
            self.svm_list.append(AbstractSvm(y, i))
        kernel = AbstractSvm.calc_kernel_matrix(x, x, sigma)
        for svm in self.svm_list:
            svm.kernel_matrix = kernel
        self.sigma = sigma
        self.c = c

    def fit(self, rate=0.001, iter_nr=1000):
        for svm in self.svm_list:
            svm.descent(self.c, rate, iter_nr)

    def predict(self, x):
        obs_kernel = AbstractSvm.calc_kernel_matrix(self.x, x, self.sigma)
        predictions = np.ndarray(shape=(len(self.svm_list), len(x)))
        for index, svm in enumerate(self.svm_list):
            svm.obs_kernel = obs_kernel
            predictions[index] = svm.predict()
        return np.argmax(predictions, axis=0)

    def check(self, x, y):
        return np.sum((self.predict(x) == y).astype(int)) * 100 / len(y)


if __name__ == '__main__':
    data = SetHolder()
    data.normalize()
    svm = MultiClassSvm(data.train_x, data.train_y)
    svm.fit()
    print(svm.check(data.test_x, data.test_y))
    