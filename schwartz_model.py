import math
import numpy as np
from process_factory import MultiDimensionItoProcess, Mu, Cov

class SchwartzModel(MultiDimensionItoProcess):
    @property
    def Sigma(self):
        return self._sigma

    @property
    def Kappa(self):
        return self._kappa

    @property
    def Theta(self):
        return self._theta

    def __init__(self, simuConfig, S0, sigma, kappa, theta):
        self._sigma = sigma
        self._kappa = kappa
        self._theta = theta
        
        def _drift(S):
            return S * self._kappa * (self._theta - math.log(S))

        def _sigma(S):
            return S * self._sigma
       
        MultiDimensionItoProcess.__init__(self, simuConfig, Mu([_drift]), Cov([[_sigma]]), [S0])

class SchwartzSmithModel:
    @property
    def Xi(self):
        return self._xi

    @property
    def Chi(self):
        return self._chi

    @property
    def Paths(self):
        return self._paths

    def __init__(self, config, S0, chi0, xi0, kappa, sigma_chi, mu_xi, sigma_xi, rho):
        self._config = config
        self._S0 = S0
        self._chi0 = chi0
        self._xi0 = xi0
        self._kappa = kappa
        self._sigma_chi = sigma_chi
        self._mu_xi = mu_xi
        self._sigma_xi = sigma_xi
        self._rho = rho
        self._nbSimus = self._config.NumberSimus
        self._nbSteps = self._config.TimeSteps
        self._dt = self._config.Dt
        self._dt_sqrt = math.sqrt(self._dt)
        self._dW_chi_t = np.zeros((self._nbSimus, self._nbSteps))
        self._dW_xi_t = np.zeros((self._nbSimus, self._nbSteps))

        for iSimu in range(0, self._nbSimus):
            for jTime in range(0, self._nbSteps):
                self._dW_xi_t[iSimu][jTime] = self._rho * self._dW_chi_t[iSimu][jTime] /\
                    - math.sqrt(1.0 - self._rho * self._rho) * self._dW_xi_t[iSimu][jTime]

        self._dW_chi_t = self._dt_sqrt * self._sigma_chi * self._dW_chi_t
        self._dW_xi_t = self._dt_sqrt * self._sigma_xi * self._dW_xi_t

    def simulate(self):
        self._paths = np.zeros((self._nbSimus, self._config.TimeSteps))
        self._chi = np.zeros((self._nbSimus, self._config.TimeSteps))
        self._xi = np.zeros((self._nbSimus, self._config.TimeSteps))

        for simu in range(0, self._nbSimus):
            self._paths[simu,0] = self._S0
            self._chi[simu,0] = self._chi0
            self._xi[simu,0] = self._xi0

        for simu in range(0, self._nbSimus):
            for time in range(1, self._nbSteps):
                self._chi[simu][time] = self._chi[simu][time-1] - self._kappa * self._chi[simu][time-1] * self._dt + self._dW_chi_t[simu][time-1]
                self._xi[simu][time] = self._xi[simu][time-1] + self._mu_xi * self._dt + self._dW_xi_t[simu][time-1]
                self._paths[simu,time] = math.exp(self._chi[simu][time] + self._xi[simu][time])

class Schwartz1FactorModel:
    @property
    def Paths(self):
        return self._paths

    @property
    def Dt(self):
        return self._dt

    @property
    def Sigma(self):
        return self._sigma


    def __init__(self, config, S0, kappa, theta, sigma):
        self._config = config
        self._S0 = S0
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma
        self._nbSimus = self._config.NumberSimus
        self._nbSteps = self._config.TimeSteps
        self._dt = self._config.Dt
        self._dt_sqrt = math.sqrt(self._dt)
        self._dW_t = np.random.normal(size =(self._nbSimus, self._nbSteps-1)) * self._dt_sqrt

    def simulate(self):
        self._paths = np.zeros((self._nbSimus, self._config.TimeSteps))
        
        for simu in range(0, self._nbSimus):
            self._paths[simu,0] = self._S0

        for simu in range(0, self._nbSimus):
            for time in range(1, self._nbSteps):
                self._paths[simu,time] = self._paths[simu,time-1] * (1.0 + self._kappa * (self._theta - math.log(self._paths[simu,time-1])) * self._dt /\
                    + self._sigma * self._dW_t[simu,time-1]) 
                    