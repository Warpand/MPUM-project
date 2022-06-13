import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from DataPreparation import SetHolder


class GradientBoost:
    def __init__(self):
        self.ensemble = None
        self.alpha = None

    def fit(self, x, y, max_depth, iter_nr):
        self.ensemble = []
        f = np.zeros(shape=(len(x), ))
        self.alpha = np.empty(shape=(iter_nr, ))
        for t in range(iter_nr):
            res = y - f
            tree = DecisionTreeRegressor(max_depth=max_depth)
            tree.fit(x, res)
            y_hat = tree.predict(x)
            step = minimize_scalar(lambda x: np.linalg.norm(y - (f + x * y_hat)), method='golden')
            a = step.x
            f += a * y_hat
            self.alpha[t] = a
            self.ensemble.append(tree)

    def predict(self, x):
        pred = np.zeros(shape=(len(x), ))
        for a, h in zip(self.alpha, self.ensemble):
            pred += a * h.predict(x)
        return np.sign(pred)

    def check(self, x, y):
        return np.sum(y == self.predict(x)) * 100 / len(y)


class OneVsOneGradientBoost:
    def __init__(self):
        self.boosters_list = []
        self.class_nr = 0

    def fit(self, x, y, max_depth=3, iter_nr=100):
        self.class_nr = len(np.unique(y))
        for i in range(self.class_nr):
            for j in range(i):
                where_i = np.argwhere(y == i).flatten()
                where_j = np.argwhere(y == j).flatten()
                booster = GradientBoost()
                idx = np.hstack([where_i, where_j])
                booster.fit(x[idx], np.hstack([np.full(len(where_i), -1, dtype=int), np.ones(len(where_j), dtype=int)]),
                            max_depth, iter_nr)
                self.boosters_list.append(booster)

    def check(self, x, y):
        counters = np.zeros(shape=(len(x), self.class_nr), dtype=int)
        index = 0
        for i in range(self.class_nr):
            for j in range(i):
                predictions = self.boosters_list[index].predict(x)
                for k, p in enumerate(predictions):
                    if p == 1:
                        counters[k][j] += 1
                    else:
                        counters[k][i] += 1
                index += 1
        # print(counters)
        return np.sum((np.argmax(counters, axis=1) == y).astype(int)) * 100 / len(y)


if __name__ == '__main__':
    data = SetHolder()
    gb = OneVsOneGradientBoost()
    gb.fit(data.train_x, data.train_y)
    print(gb.check(data.test_x, data.test_y))
