import numpy as np
import math
import matplotlib.pyplot as plt
import time


class SimulationConfig:
    @property
    def T(self):
        return self._T

    @property
    def NumberSimus(self):
        return self._numberSimus

    @property
    def TimeSteps(self):
        return self._timeSteps

    @property
    def Dt(self):
        return self._T / self._timeSteps

    def __init__(self, T, numberSimus, timeSteps):
        self._T = T
        self._numberSimus = numberSimus
        self._timeSteps = timeSteps
        self._dt = .0

class Mu:
    @property
    def Entries(self):
        return self._entries

    def __init__(self, entries):
        self._entries = entries

    def __call__(self, s):                
        assert len(self._entries) == len(s), 'x is of wrong length!'
        l = len(s)
        ret = np.zeros(l)

        for i in range(0, l):
            ret[i] = self._entries[i](s)

        return ret

class Cov:
    def __init__(self, cov):
        self._cov = cov

    def __call__(self, x, dW):            
        assert len(self._cov) == len(x), 'x is of wrong length!'
        l = len(x)
        ret = np.zeros(l)

        for i in range(0, l):
            v = 0
            for j in range(0, l):
                v += self._cov[i][j](x) * dW[j]
            ret[i] = v

        return ret

class Const:
    def  __init__(self, value):
        self._value = value

    def __call__(self, x):
        return self._value

class Identity:
    def __call__(self, x):
        return x

class MultiDimensionItoProcess:
    @property
    def NbSimus(self):
        return self._nbSimus

    @property
    def Dim(self):
        return self._dim

    @property
    def SimuConfig(self):
        return self._simuConfig

    @property
    def S0(self):
        return self._S0

    @property
    def NbSteps(self):
        return self._nbSteps

    @property
    def Cov(self):
        return self._cov

    @property
    def Mu(self):
        return self._mu

    @property
    def DW_t(self):
        return self._dW_t

    @property
    def Dt(self):
        return self._dt

    def SetProgressBar(self, progress_bar):
        self._progress_bar = progress_bar

    def SetManager(self, manager):
        self._manager = manager

    def __init__(self, simuConfig, mu, cov, S0):
        self._simuConfig = simuConfig      
        self._mu = mu
        self._cov = cov
        self._dim = len(mu.Entries)        
        self._dW_t = np.zeros((self._dim, self._simuConfig.NumberSimus, self._simuConfig.TimeSteps))
        self._S0 = S0
        self._dt = .0
        self._nbSimus = self._simuConfig.NumberSimus
        self._nbSteps = self._simuConfig.TimeSteps
        self.initialize()
    
    def generatePaths(self):
        paths = np.zeros((self._dim, self._nbSimus, self._simuConfig.TimeSteps))

        for simu in range(0, self._nbSimus):
            paths[:,simu,0] = self._S0

        for simu in range(0, self._nbSimus):
            for time in range(1, self._nbSteps):
                paths[:,simu,time] = paths[:,simu,time-1] + self._cov(paths[:,simu,time-1], self._dW_t[:,simu,time-1])
                paths[:,simu,time] = paths[:,simu,time] + self._mu(paths[:,simu,time-1]) * self._dt

        return paths

    def initialize(self):
        dt = self._simuConfig.Dt
        self._dt = dt
        dt_sqrt = math.sqrt(dt)
       
        self._dW_t = np.random.normal(size =(self._dim, self._nbSimus, self._nbSteps)) * dt_sqrt







     




  













