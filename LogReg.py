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
        :return: sigmoid function of x"""
        return 1/(1-np.exp(-x))

    def train(self, X, y, alpha=0.01, max_num_iters=150):
        """
        Train logistic regression
        :type X: np.double
        :type y: np.double
        :type alpha: float
        :type max_num_iters: int
        :param X: learning examples
        :param y: target
        :param alpha: learning rate
        :param max_num_iters: max number of iteration of gradient descent
        """
        m = y.size
        self.__theta = np.random(m)
        J = np.iinfo(np.float64).max #costFunction

        for i in range(max_num_iters):
            h = np.dot(X, self.__theta)
            theta = theta - alpha/m*(np.dot((h - y).T, X)).T
            J_new = 1/(2*m)*np.sum(np.power(h-y, 2))
            if J_new > J:
                raise Exception("Wrong learning rate method diverge")
            J = J_new

    def predict(self, X, threshold):
        """
        :type X: np.double
        :param X: predicted examples
        :type threshold: float
        :param threshold: threshold
        :returns: predicted value
        :rtype: np.double"""
