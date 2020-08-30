import math
from process_factory import *
from schwartz_model import *
from calibrators.max_likelihood import MaxLikelihood
from calibrators.least_square import LeastSquare


simuConfig = SimulationConfig(2.0, 1000, 800)
S0 = 1.0
kappa = .1
theta = .2
sigma = .3

schwartz_one_factor = Schwartz1FactorModel(simuConfig, S0, kappa, theta, sigma)
schwartz_one_factor.simulate()
paths = schwartz_one_factor.Paths

max_likelihood = MaxLikelihood(paths[0], schwartz_one_factor)

leas_square = LeastSquare(paths[0], schwartz_one_factor)