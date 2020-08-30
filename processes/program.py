import math
from process_factory import *
from schwartz_model import SchwartzSmithModel


simuConfig = SimulationConfig(2.0, 1, 800)
S0 = 1.0
kappa = .1
theta = .2
sigma = .3

schwartz_one_factor = SchwartzOneFactorModel(simuConfig, S0, theta, sigma)
schwartz_smith.simulate()


