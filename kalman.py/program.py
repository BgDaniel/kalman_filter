import numpy as np
import math
import matplotlib.pyplot as plt
from dynamic_system import DynamicSystem
from linear_kalman_filter import KalmanSchwartzFilter

x0 = np.array([2.0, .0])
nb_dims = len(x0)
dphi = 2.0 * math.pi * .01
X = np.array([[math.cos(dphi), - math.sin(dphi)], [math.sin(dphi), math.cos(dphi)]])
SQUARE_ROOT_TWO = math.sqrt(2.0)
Z = np.array([[1.0 / SQUARE_ROOT_TWO, 1.0 / SQUARE_ROOT_TWO], [1.0 / SQUARE_ROOT_TWO, -1.0 / SQUARE_ROOT_TWO]])

cov_X = np.array(([[.015, -.01], [-.01, 0.015]]))
cov_Z = np.array(([[.02, +.01], [+.01, 0.015]]))

system = DynamicSystem(x0, X, Z, cov_X, cov_Z)
nb_times = 100
x, z = system.simulate(nb_times)

#plt.plot(z[:,0], z[:,1])
#plt.show()

#plt.plot(x[:,0], x[:,1])
#plt.show()

kalman_filter = KalmanSchwartzFilter(system)
x_hat, P, K, x_hat_0 = kalman_filter.filter(z)

#plot estimate of hidden variable path compare to real value
#plt.plot(x[:,0], x[:,1])
#plt.plot(x_hat[:,0], x_hat[:,1])
#plt.show()

nb_simus = 1000
X = np.zeros((nb_simus,nb_times,nb_dims))
X_hat = np.zeros((nb_simus,nb_times,nb_dims))
X_hat_0 = np.zeros((nb_simus,nb_times,nb_dims))

for iSimu in range(0, nb_simus):
    X[iSimu], Z = system.simulate(nb_times)
    X_hat[iSimu], P, K, X_hat_0[iSimu] = kalman_filter.filter(Z)

#create statistics
mean_X = np.zeros((nb_times,nb_dims))
mean_X_hat = np.zeros((nb_times,nb_dims))
mean_X_hat_0 = np.zeros((nb_times,nb_dims))

for iTime in range(0, nb_times):
    for iDim in range(0, nb_dims):
        for iSimu in range(0, nb_simus):
            mean_X[iTime,iDim] += X[iSimu,iTime,iDim] / float(nb_simus)
            mean_X_hat[iTime,iDim] += X_hat[iSimu,iTime,iDim] / float(nb_simus)
            mean_X_hat_0[iTime,iDim] += X_hat_0[iSimu,iTime,iDim] / float(nb_simus)

plt.plot(mean_X[:,0], mean_X[:,1], color='red')
plt.plot(mean_X_hat[:,0], mean_X_hat[:,1], color='green')
plt.plot(mean_X_hat_0[:,0], mean_X_hat_0[:,1], color='yellow')
plt.show()

stdev_X_hat = np.zeros((nb_times,nb_dims))
stdev_X_hat_0 = np.zeros((nb_times,nb_dims))

for iTime in range(0, nb_times):
    for iDim in range(0, nb_dims):
        for iSimu in range(0, nb_simus):
            stdev_X_hat[iTime,iDim] += (X_hat[iSimu,iTime,iDim] - X[iSimu,iTime,iDim]) * (X_hat[iSimu,iTime,iDim] - X[iSimu,iTime,iDim]) / float(nb_simus-1)
            stdev_X_hat_0[iTime,iDim] += (X_hat_0[iSimu,iTime,iDim] - X[iSimu,iTime,iDim]) * (X_hat_0[iSimu,iTime,iDim] - X[iSimu,iTime,iDim]) / float(nb_simus-1)

        stdev_X_hat[iTime,iDim] = math.sqrt(stdev_X_hat[iTime,iDim])
        stdev_X_hat_0[iTime,iDim] = math.sqrt(stdev_X_hat_0[iTime,iDim])

for iDim in range(0, nb_dims):
    plt.plot(stdev_X_hat[:,iDim], color='red')
    plt.plot(stdev_X_hat_0[:,iDim], color='green')
    plt.ylim(.0, max(np.max(stdev_X_hat[:,iDim]), np.max(stdev_X_hat_0[:,iDim]))) 
    plt.show()