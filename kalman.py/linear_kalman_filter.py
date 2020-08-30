import numpy as np

class KalmanSchwartzFilter:
    def __init__(self, dynamic_system):
        self._x0 = dynamic_system.X0
        self._dim = dynamic_system.Dim
        self._X = dynamic_system.X
        self._Z = dynamic_system.Z
        self._cov_X = dynamic_system.Cov_X
        self._cov_Z = dynamic_system.Cov_Z

    def _update(z_k, x_estimate):
        return x_estimate_updated

    def filter(x, z):
        nb_steps = len(path)

        # randomnly initialize estimate of x for t=0        
        x_estimate = np.zeros((self._dim))

        for iStep in range(0, nb_steps):
            z_k = z[iStep]
            x_estimate = self._update(z_k, x_estimate)