import math
from process_factory import *
from schwartz_model import *
from calibrators.max_likelihood import MaxLikelihood
from calibrators.least_square import LeastSquare


simuConfig = SimulationConfig(10.0, 1, 10000)
S0 = 1.0
kappa = 12.0
theta = 1.0
sigma = .3

schwartz_one_factor = Schwartz1FactorModel(simuConfig, S0, kappa, theta, sigma)
schwartz_one_factor.simulate()
paths = schwartz_one_factor.Paths

max_likelihood = MaxLikelihood(paths[0], schwartz_one_factor)
#kappa, theta = max_likelihood.calibrate()

theta = max_likelihood.calibrate_theta()
print(theta)

kappa = max_likelihood.calibrate_kappa()
print(kappa)

least_square = LeastSquare(paths[0], schwartz_one_factor)
kappa, theta = least_square.calibrate()