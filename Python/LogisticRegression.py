import numpy as np


class SoftmaxRegression:
    @staticmethod
    def class_encode(y, c):
        y_enc = np.zeros((y.shape[0], c))
        y_enc[np.arange(y.shape[0]), y] = 1

        return y_enc

    @staticmethod
    def softmax(z):
        exp = np.exp(z-np.max(z))

        for i in range(z.shape[0]):
            exp[i] /= np.sum(exp[i])

        return exp

    def fit(self, X_train, y_train, eta, classes, iterations):
        m, n = X_train.shape

        self.theta = np.random.random((n, classes))
        self.b = np.random.random(classes)

        log_losses = []

        for iter in range(iterations):

            z = X_train @ self.theta + self.b
            y_pred = self.softmax(z)

            y_enc = self.class_encode(y_train, classes)

            grad_b = (1 / m) * np.sum(y_pred - y_enc)
            grad_theta = (1 / m) * np.dot(X_train.T, (y_pred - y_enc))

            self.theta = self.theta - eta * grad_theta
            self.b = self.b - eta * grad_b

            log_loss = -np.mean(np.log(y_pred[np.arange(y_train.shape[0]), y_train]))
            log_losses.append(log_loss)

            if iter % 1000 == 0:
                print(f'Iter {iter}==> Loss = {log_loss}')

    def predict(self, X_test):
        z = X_test @ self.theta + self.b
        y_pred = self.softmax(z)

        return np.argmax(y_pred, axis=1)

    def check(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_test == y_pred) / y_test.shape[0]
