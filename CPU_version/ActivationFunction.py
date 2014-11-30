# -*-coding:Utf-8 -*

import numpy as np


class Sigmoid(object):

    def function(self, x):
        return (1 / (1 + np.exp(-x)))

    def derivate(self, x):
        return np.multiply(self.function(x),1 - self.function(x))


class Tanh(object):

    def function(self, x):
        return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    def derivate(self, x):
        return 1 - np.multiply(self.function(x),self.function(x))

class RectifiedLinear(object):

    def __init__(self, slope_rupt):
        self.slope_rupt = float(slope_rupt[0])

    def function(self,x):
        x[x<= self.slope_rupt] = 0
        x[x > self.slope_rupt] = (x[x >self.slope_rupt] - self.slope_rupt) / (1. - self.slope_rupt)
        x[x > 1 ] = 1

        return x

    def derivate(self,x):
        x[x<= self.slope_rupt] = 0
        x[x > self.slope_rupt] = 1 / (1 - self.slope_rupt)

        return x

def test():
    slope_rupt = 0.2

    sigmoid = Sigmoid()
    rect_lin = RectifiedLinear(slope_rupt)

    print sigmoid.function(0.5)
    print sigmoid.derivate(0.5)

    print rect_lin.function(0.2)
    print rect_lin.derivate(0.2)

if __name__ == "__main__":
    test()

