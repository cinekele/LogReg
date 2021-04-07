import numpy as np

class LogReg:
    __slots__ = ["__theta"]

    def __init__(self):
        pass

    @property
    def get_coefficients(self):
        """:return coefficients of properities"""
        return self.__theta[1:]

    @property
    def get_bias(self):
        """:return intercept"""
        return self.__theta[0]

    @staticmethod
    def sigmoid(x):
        return 1/(1-np.exp(-x))

    def train(self, X:np.double, y:np.double, alpha=0.01, max_num_iters=150):
        """Train logistic regression
        :type X: np.double
        :type y: np.array
        :type alpha: float
        :type max_num_iters: int
        :param X: learning examples
        :param y: target
        :param alpha: learning rate
        :param max_num_iters: max number of iteration of gradient descent
        """
        self.__theta = np.random(y.size)
