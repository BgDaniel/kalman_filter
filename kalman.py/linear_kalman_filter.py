import numpy as np

class KalmanSchwartzFilter:
    def __init__(self, dynamic_system):
        self._x0 = dynamic_system.X0
        self._dim = dynamic_system.Dim
        self._X = dynamic_system.X
        self._X_T = self._X.transpose()
        self._Z = dynamic_system.Z
        self._Z_T = self._Z.transpose()
        self._cov_X = dynamic_system.Cov_X
        self._cov_Z = dynamic_system.Cov_Z

    def _compute_K(self, P):
        A = P.dot(self._Z_T)
        B = self._Z.dot(P.dot(self._Z_T)) + self._cov_Z
        C = np.linalg.inv(B)
        return A.dot(C)

    def _compute_P(self, K, P):
        A = np.identity(self._dim) - K.dot(self._Z)
        return A.dot(P)
    
    def _update(self, z, x_hat, P, step):
        K = self._compute_K(P)

        trace_before = np.trace(P)
        P = self._compute_P(K, P)
        trace_after = np.trace(P)
        reduction_trace = 100.0 * (trace_after - trace_before) / trace_before

        #print('Trace reduced by {0} percent in step {1}.'.format(reduction_trace, step))     

        x_hat = x_hat + K.dot(z - self._Z.dot(x_hat))
        return x_hat, P, K

    def filter(self, z):
        nb_steps = len(z)
        x_hat = np.zeros((nb_steps,self._dim))
        x_hat_0 = np.zeros((nb_steps,self._dim))
        P = np.zeros((nb_steps,self._dim,self._dim))
        K = np.zeros((nb_steps,self._dim,self._dim))

        # initialize estimate of x for t=0        
        x_hat[0] = self._x0
        x_hat_0[0] = self._x0
        P[0] = np.zeros((self._dim,self._dim))

        for iStep in range(0, nb_steps-1):
            # update x_hat and P based on realization of non hidden variable z_k
            x_hat[iStep], P[iStep], K[iStep] = self._update(z[iStep], x_hat[iStep], P[iStep], iStep+1)

            # translate to following time step
            x_hat[iStep+1] = self._X.dot(x_hat[iStep])
            x_hat_0[iStep+1] = self._X.dot(x_hat_0[iStep])
            P[iStep+1] = self._X.dot(P[iStep].dot(self._X_T)) + self._cov_X

        return x_hat, P, K, x_hat_0


