import numpy as np

class LogReg:
    __slots__ = ["__theta"]

    def __init__(self):
        pass

    @property
    def get_coefficients(self):
        """Return coefficients of properities"""
        return self.__theta[1:]

    @property
    def get_bias(self):
        """Return intercept"""
        return self.__theta[0]

    def train(self, X, y):
        """"""