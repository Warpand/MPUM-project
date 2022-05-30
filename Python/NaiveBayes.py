import numpy as np


class GaussNaiveBayes:

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.y_probs = (np.bincount(y_train)+1)/(y_train.shape[0]+8)

        self.μ = np.array([X_train[np.where(y_train == i)].mean(axis=0) for i in self.classes])
        self.σ = np.array([X_train[np.where(y_train == i)].std(axis=0) for i in self.classes])

    def predict(self, X_test):
        result = []
        for i in range(X_test.shape[0]):
            probabilities = []
            for j in range(self.classes.shape[0]):
                prob = np.log(self.y_probs[j])
                for ii in range(X_test.shape[1]):
                    if self.σ[j][ii] == 0:
                        continue
                    prob += np.log(1 / np.sqrt(2 * np.pi * self.σ[j][ii] ** 2)) - 0.5 * ((X_test[i][ii] - self.μ[j][ii]) / self.σ[j][ii]) ** 2
                probabilities.append(prob)
            result.append(np.array(probabilities))

        result = np.array(result)
        result = result.argmax(axis=1)
        result = [self.classes[result[i]] for i in range(result.shape[0])]

        return np.array(result)

    def check(self, X_test, y_test):
        classified = 0

        y_pred = self.predict(X_test)

        for i in range(y_pred.shape[0]):
            if y_test[i] == y_pred[i]:
                classified += 1
        return classified/y_pred.shape[0]


class MultNaiveBayes:
    def fit(self, X_train, y_train, argmax):
        self.classes = np.unique(y_train)
        count_y = np.bincount(y_train)
        self.phi_y = (np.bincount(y_train)+1)/(y_train.shape[0]+8)
        self.params = np.array([np.apply_along_axis(
            lambda x: np.bincount(x, minlength=argmax), axis=0, arr=X_train[np.where(y_train == i)]) for i in self.classes]) + 1
        self.params /= 10
        for i in range(count_y.shape[0]):
            self.params[i] = 10*self.params[i]/(count_y[i]+argmax)

    def predict(self, X_test):
        result = []
        for i in range(X_test.shape[0]):
            probabilities = []
            for j in range(self.classes.shape[0]):
                prob = np.log(self.phi_y[j])
                for ii in range(X_test.shape[1]):
                    prob += np.log(self.params[j][X_test[i][ii]][ii])
                probabilities.append(prob)
            result.append(np.array(probabilities))

        result = np.array(result)
        result = result.argmax(axis=1)
        result = [self.classes[result[i]] for i in range(result.shape[0])]

        return np.array(result)

    def check(self, X_test, y_test):
        classified = 0

        y_pred = self.predict(X_test)

        for i in range(y_pred.shape[0]):
            if y_test[i] == y_pred[i]:
                classified += 1
        return classified/y_pred.shape[0]