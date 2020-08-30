import math
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from calibrators.calibrator_base import CalibratorBase

class LeastSquare(CalibratorBase):
    def __init__(self, path, schwartz_model):
        CalibratorBase.__init__(self, path, schwartz_model)
        self._nbSamples = len(path) - 1
        self._dt = schwartz_model.Dt

    def calibrate(self):
        X = np.zeros((self._nbSamples,1))
        y = np.zeros((self._nbSamples))

        for i in range(0, self._nbSamples):
            X[i][0] = math.log(self._path[i])
            y[i] = math.log(self._path[i + 1])

        reg = LinearRegression().fit(X, y)
        alpha, beta = reg.coef_[0], reg.intercept_

        theta = beta / (1.0 - alpha)
        kappa = (1.0 - alpha) / self._dt
        
        return theta, kappa