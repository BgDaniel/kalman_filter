import numpy as np
import math
import matplotlib.pyplot as plt
from dynamic_system import DynamicSystem
from linear_kalman_filter import KalmanSchwartzFilter

x0 = np.array([2.0, .0])
X = np.array([[.0, - 1.0], [1.0, .0]])
SQUARE_ROOT_TWO = math.sqrt(2.0)
Z = np.array([[1.0 / SQUARE_ROOT_TWO, 1.0 / SQUARE_ROOT_TWO], [1.0 / SQUARE_ROOT_TWO, -1.0 / SQUARE_ROOT_TWO]])

cov_X = np.array(([[.01, -.03], [-.03, 0.06]]))
cov_Z = np.array(([[.08, +.04], [+.04, 0.03]]))

system = DynamicSystem(x0, X, Z, cov_X, cov_Z)
x, z = system.simulate(100)

#plt.plot(z[:,0], z[:,1])
#plt.show()

kalman_filter = KalmanSchwartzFilter(system)
kalman_filter.filter()

