import math
from scipy.stats import norm
from calibrators.calibrator_base import CalibratorBase
import scipy.optimize as opt

class MaxLikelihood(CalibratorBase):
    def __init__(self, path, schwartz_model):
        CalibratorBase.__init__(self, path, schwartz_model)
        self._nbSamples = len(path) - 1
        self._sigma = math.sqrt(schwartz_model.Dt) * schwartz_model.Sigma
        self._dt = schwartz_model.Dt
        self._kappa = schwartz_model.Kappa
        self._theta = schwartz_model.Theta

    def _theta(self, kappa):
        val = .0

        for i in range(0, self._nbSamples):
            val += math.log(self._path[i + 1] / self._path[i]) / self._dt + kappa * math.log(self._path[i])

        return val / (float(self._nbSamples) * kappa)

    def _d_kappa(self, kappa):
        theta = self._theta(kappa)

        if True:
            self.cross_check(kappa, theta)

        d_kappa = .0

        for i in range(0, self._nbSamples):
            x_i = math.log(self._path[i + 1] / self._path[i]) - kappa * (theta - math.log(self._path[i])) * self._dt
            x_i *= math.log(self._path[i])
            d_kappa += x_i

        return d_kappa

    def cross_check(self, kappa, theta):
        val_kappa, val_theta = .0, .0
        kappa = kappa[0]

        for i in range(0, self._nbSamples):
            x_i = math.log(self._path[i + 1] / self._path[i]) - kappa * (theta - math.log(self._path[i])) * self._dt
            val_kappa += (x_i * (theta - kappa * math.log(self._path[i])))
            val_theta += (kappa * x_i)

        assert abs(val_kappa) < 1e-8, 'Error in kappa equation!'
        assert abs(val_theta) < 1e-14, 'Error in theta equation!'


    def calibrate(self): 
        kappa0 = .3

        sol = opt.root(self._d_kappa, kappa0)

        if sol.success:
            kappa, theta = sol.x[0], self._theta(sol.x[0])
            return True, kappa, theta

        return False, None

    def calibrate_theta(self): 
        theta = .0

        for i in range(0, self._nbSamples):
            theta += math.log(self._path[i + 1] / self._path[i]) + math.log(self._path[i]) * self._kappa * self._dt

        theta /= (self._nbSamples * self._kappa * self._dt)

        if True:
            cross_check = .0

            for i in range(0, self._nbSamples):
                cross_check +=  math.log(self._path[i + 1] / self._path[i]) - self._kappa * (theta - math.log(self._path[i])) * self._dt

            assert abs(cross_check) < 1e-10, 'Error in calibrate_theta!'

        return theta

    def calibrate_kappa(self): 
        kappa = .0

        nom, denom = .0, .0        

        for i in range(0, self._nbSamples):
            nom += (self._theta - math.log(self._path[i])) * math.log(self._path[i + 1] / self._path[i])
            denom += (self._theta - math.log(self._path[i])) * (self._theta - math.log(self._path[i]))

        denom *= self._dt
        kappa = nom / denom

        if True:
            cross_check = .0

            for i in range(0, self._nbSamples):
                cross_check += (math.log(self._path[i + 1] / self._path[i]) - kappa * (self._theta - math.log(self._path[i])) * self._dt) *\
                    (self._theta - math.log(self._path[i]))

            assert abs(cross_check) < 1e-10, 'Error in calibrate_theta!'

        return kappa

        


