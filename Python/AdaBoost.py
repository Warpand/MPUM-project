import numpy as np

from DataPreparation import SetHolder
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self):
        self.alphas = None
        self.h = []
        self.class_nr = 0

    def fit(self, x, y, iter_nr=50):
        self.h = []
        self.alphas = np.empty(shape=(iter_nr, ))
        self.class_nr = len(np.unique(y))
        weights = np.full((len(x), ), 1 / len(x))
        for t in range(iter_nr):
            st = DecisionTreeClassifier(max_depth=1)
            st.fit(x, y, sample_weight=weights)
            y_pred = st.predict(x)
            err = np.sum(weights * np.not_equal(y, y_pred).astype(int))
            alpha = np.log((1.0 - err) / err) + np.log(self.class_nr - 1)
            self.alphas[t] = alpha
            weights *= np.exp(alpha * np.not_equal(y, y_pred).astype(int))
            weights /= np.sum(weights)
            self.h.append(st)

    def _predict(self, obs):
        prediction = 0
        max_val = 0.0
        for p in range(self.class_nr):
            val = 0.0
            for stump, alpha in zip(self.h, self.alphas):
                if stump.predict(obs.reshape(1, -1))[0] == p:
                    val += alpha
            if val > max_val:
                max_val = val
                prediction = p
        return prediction

    def check(self, x, y):
        y_pred = np.empty(shape=(len(x), ), dtype=int)
        for i, val in enumerate(x):
            y_pred[i] = self._predict(val)
        return np.sum(np.equal(y, y_pred).astype(int)) / len(y) * 100.0


if __name__ == '__main__':
    data = SetHolder()
    ad = AdaBoost()
    ad.fit(data.train_x, data.train_y, 100)
    ac = ad.check(data.test_x, data.test_y)
    print(ac)
