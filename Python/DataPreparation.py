import numpy as np

TRAIN_SIZE = 6400
VAL_SIZE = 800
TEST_SIZE = 800
FEATURE_NR = 518

TRAIN_FILE = '../train.data'
VAL_FILE = '../val.data'
TEST_FILE = '../test.data'


def read_data(x, y, filename):
    with open(filename, 'r') as file:
        for line_nr, line in enumerate(file):
            numbers = line.split()
            for i in range(FEATURE_NR):
                x[line_nr][i] = float(numbers[i])
            y[line_nr] = int(numbers[FEATURE_NR]) - 1


class SetHolder:
    def __init__(self):
        self.train_x = np.ndarray(shape=(TRAIN_SIZE, FEATURE_NR))
        self.train_y = np.empty(TRAIN_SIZE, dtype=int)
        self.val_x = np.ndarray(shape=(VAL_SIZE, FEATURE_NR))
        self.val_y = np.empty(VAL_SIZE, dtype=int)
        self.test_x = np.ndarray(shape=(TEST_SIZE, FEATURE_NR))
        self.test_y = np.empty(TEST_SIZE, dtype=int)
        read_data(self.train_x, self.train_y, TRAIN_FILE)
        read_data(self.val_x, self.val_y, VAL_FILE)
        read_data(self.test_x, self.test_y, TEST_FILE)

    def standardize(self):
        stds = np.std(self.train_x, axis=0)
        means = np.mean(self.train_x, axis=0)
        self.train_x -= means
        self.train_x /= stds
        self.val_x -= means
        self.val_x /= stds
        self.test_x -= means
        self.test_x /= stds

    def normalize(self):
        minimums = np.min(self.train_x, axis=0)
        difference = np.max(self.train_x, axis=0) - minimums
        self.train_x -= minimums
        self.train_x /= difference
        self.val_x -= minimums
        self.val_x /= difference
        self.test_x -= minimums
        self.test_x /= difference

    @staticmethod
    def _pca(x, dim, threshold):
        x = x - np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)
        eig_values, eig_vectors = np.linalg.eigh(cov)
        idx = np.argsort(eig_values)
        sorted_vectors = eig_vectors[:, idx]
        if dim is None:
            sorted_values = eig_values[idx]
            prefix = np.cumsum(sorted_values) / np.sum(sorted_values)
            print(prefix)
            for index, val in enumerate(prefix):
                if val >= threshold:
                    dim = index + 1
                    break
        return (sorted_vectors[:, :dim].transpose() @ x.transpose()).transpose()

    def pca(self, dim=None, threshold: float = 0.95):
        """ Performs principal component analysis to lower the dimension
            of data to the one given in parameter dim. When dim is None
            it's calculated based on threshold parameter (iif dim is not None
            threshold is ignored) Threshold should be between 0.0 and 100.0.
            This method should be called before calling normalize or standardize."""
        stacked_data = np.vstack([self.train_x, self.val_x, self.test_x])
        new_data = SetHolder._pca(stacked_data, dim, threshold)
        self.train_x = new_data[:TRAIN_SIZE]
        self.val_x = new_data[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
        self.test_x = new_data[TRAIN_SIZE + VAL_SIZE:]
