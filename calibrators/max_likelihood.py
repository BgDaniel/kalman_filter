import math
from scipy.stats import norm
from calibrators.calibrator_base import CalibratorBase

class MaxLikelihood(CalibratorBase):
    def __init__(self, path, schwartz_model):
        CalibratorBase.__init__(self, path, schwartz_model)
        self._nbSamples = len(path) - 1
        self._locVol = math.sqrt(schwartz_model.Dt) * schwartz_model.Sigma

    def ell(self, kappa, theta):
        val_kappa = .0
        val_theta = .0

        for i in range(0, self._nbSamples):
            x_i = math.log(self._path[i + 1]) - math.log(self._path[i]) - kappa * (theta - math.log(self._path[i]))
            dphi_dx_i = (- x_i / (self._locVol * self._locVol)) * math.log(norm.pdf(.0, self._locVol))(x_i)
            val_kappa += (- theta + math.log(self._path[i])) * dphi_dx_i
            val_theta += - kappa * dphi_dx_i

        return val_kappa, val_theta

    def calibrate(self):        
        def _ell(kappa, theta):
            val_kappa = .0
            val_theta = .0

            for i in range(0, self._nbSamples):
                x_i = math.log(self._path[i + 1] - self._path[i]) - kappa * (theta - math.log(self._path[i]))
                dphi_dx_i = (- x_i / (self._locVol * self._locVol)) * math.log(norm.pdf(.0, self._locVol))(x_i)
                val_kappa += (- theta + math.log(self._path[i])) * dphi_dx_i
                val_theta += - kappa * dphi_dx_i

            return val_kappa, val_theta
