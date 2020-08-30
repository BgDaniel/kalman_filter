import math
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from calibrator_base import CalibratorBase

class LeastSquare(CalibratorBase):
    def __init__(self, path, schwartz_model):
        CalibratorBase.__init__(self, path)
        self._nbSamples = len(path) - 1

    def calibrate(self):
        X = np.zeros((self._nbSamples))
        y = np.zeros((self._nbSamples))

        for i in range(0, self._nbSamples):
            X[i] = math.log(self._path[i] + 1)
            y[i] = math.log(self._path[i + 1])

        reg = LinearRegression().fit(X, y)
        alpha, beta = reg.coef_[0], reg.coef_[1]

        theta = beta / (1.0 - alpha)
        kappa = (1.0 - alpha) / schwartz_model.Dt
        
        return kappa, theta