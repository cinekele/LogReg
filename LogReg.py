import numpy as np

class LogReg:
    __slots__ = ["__theta"]

    def __init__(self):
        pass

    @property
    def get_coefficients(self):
        """
        :returns: coefficients of properities
        :rtype: np.double
        """
        return self.__theta[1:]

    @property
    def get_bias(self):
        """
        :return: intercept
        :rtype: np.double
        """
        return self.__theta[0]

    @staticmethod
    def sigmoid(x):
        """
        :return: sigmoid function of x
        :rtype: np.double
        """
        return 1/(1-np.exp(-x))

    def train(self, X, y, alpha=0.01, max_num_iters=150, tol=1e-4, reg_term=0.01):
        """
        Train logistic regression using gradient decent

        :param X: learning examples
        :type X: np.double

        :param y: target
        :type y: np.double

        :param alpha: learning rate
        :type alpha: float

        :param max_num_iters: max number of iteration of gradient descent
        :type max_num_iters: int

        :param tol: tolerance of converging
        :type tol: float

        :param reg_term: regularization term
        :type reg_term: float
        """
        m = y.size
        self.__theta = np.random(m)
        J = np.iinfo(np.float64).max #costFunction
        X_bias = np.append(np.zeros((1,m)), X)

        for i in range(max_num_iters):
            h = np.dot(X_bias, self.__theta)
            self.__theta -= alpha/m*(np.dot((h - y).T, X)).T + reg_term/m*(X)
            # J_new = 1/(2*m)*(np.sum(np.power(h-y, 2))+np.sum(np.power()))
            # if J_new > J:
            #     raise Exception("Wrong learning rate method diverge")
            # if np.abs(J_new - J) < tol:
            #     break
            # J = J_new

    def predict(self, X, threshold=0.5):
        """
        Predict based on given values

        :type X: np.double
        :param X: predicted examples

        :type threshold: float
        :param threshold: threshold

        :returns: predicted value
        :rtype: np.double
        """

        return self.sigmoid(np.dot(X, self.__theta)) >= threshold

    def predict_proba(self, X):
        """
        :type X: np.double
        :param X: predicted examples

        :returns: probabilities of values
        :rtype: np.double
        """

        return self.sigmoid(np.dot(X, self.__theta))