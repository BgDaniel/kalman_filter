import numpy as np

class DynamicSystem:
    @property
    def X0(self):
        return self._x0

    @property
    def Dim(self):
        return self._dim

    @property
    def X(self):
        return self._X

    @property
    def Z(self):
        return self._Z

    @property
    def Cov_Z(self):
        return self._cov_Z

    @property
    def Cov_X(self):
        return self._cov_X

    def __init__(self, x0, X, Z, cov_X, cov_Z):
        self._x0 = x0
        self._dim = len(x0)
        self._X = X
        self._Z = Z
        self._cov_X = cov_X
        self._cov_Z = cov_Z

    def simulate(self, nb_steps):
        x = np.zeros((nb_steps, self._dim))
        z = np.zeros((nb_steps, self._dim))

        x[0] = self._x0
        
        zero_mean = np.zeros((self._dim))
        noise_x = np.random.multivariate_normal(zero_mean, self._cov_X, nb_steps)
        noise_z = np.random.multivariate_normal(zero_mean, self._cov_Z, nb_steps)
        
        for iStep in range(1, nb_steps):
            z[iStep-1] = self._Z.dot(x[iStep-1]) + noise_z[iStep-1]
            x[iStep] = self._X.dot(x[iStep-1]) + noise_x[iStep]

        return x, z

     