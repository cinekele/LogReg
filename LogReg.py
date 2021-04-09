from numpy import concatenate, abs, log, dot, ones, finfo, copy, float64, zeros
from warnings import warn
from scipy.special import expit  # sigmoid function


class LogReg:
    __slots__ = ["__theta", "alpha", "max_num_iters", "tol", "reg_term"]

    def __init__(self, alpha=1e-4, max_num_iters=150, tol=1e-4, reg_term=0.0001):
        """
        param alpha: learning rate
        :type alpha: float

        :param max_num_iters: max number of iteration of gradient descent
        :type max_num_iters: int

        :param tol: tolerance of converging
        :type tol: float

        :param reg_term: regularization term
        :type reg_term: float
        """
        self.alpha = alpha
        self.max_num_iters = max_num_iters
        self.tol = tol
        self.reg_term = reg_term

    @property
    def get_coefficients(self):
        """
        :returns: coefficients of properities
        :rtype: np.double
        """
        return copy(self.__theta[1:])

    @property
    def get_bias(self):
        """
        :return: intercept
        :rtype: np.double
        """
        return copy(self.__theta[0])

    def train(self, X, y):
        """
        Train logistic regression using gradient decent

        :param X: learning examples
        :type X: np.array

        :param y: target
        :type y: np.array
        """
        m = X.shape[0]
        n = X.shape[1]
        self.__theta = zeros(n + 1)
        j = finfo(float64).max  # cost function

        X_bias = concatenate((ones((m, 1)), X), axis=1)
        offset = 1e-6

        for i in range(self.max_num_iters):
            h = expit(dot(X_bias, self.__theta))
            loss = h - y

            without_bias = copy(self.__theta)
            without_bias[0] = 0  # don't penalize bias

            grad = self.alpha * (dot(loss.T, X_bias)).T / m  # real gradient
            reg_grad = self.reg_term * without_bias / m  # regularization term
            self.__theta -= grad + reg_grad

            jcost = (-1 / m) * (dot(log(h + offset).T, y) + dot(log((ones(m) - h + offset)).T, ones(m) - y))
            jreg = self.reg_term / (2 * m) * dot(without_bias.T, without_bias)
            j_new = jcost + jreg

            if abs(j_new - j) < self.tol:
                break
            j = j_new

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
        m = X.shape[0]
        X_bias = concatenate((ones((m, 1)), X), axis=1)
        return expit(dot(X_bias, self.__theta)) >= threshold

    def predict_proba(self, X):
        """
        :type X: np.double
        :param X: predicted examples

        :returns: probabilities of values
        :rtype: np.double
        """
        m = X.shape[0]
        X_bias = concatenate((ones((m, 1)), X), axis=1)
        return expit(dot(X_bias, self.__theta))


