import math
from process_factory import *
from schwartz_model import *
from calibrators.max_likelihood import MaxLikelihood
from calibrators.least_square import LeastSquare

nb_simus = 100

simuConfig = SimulationConfig(5.0, nb_simus, 5000)
S0 = 1.0
kappa = 12.0
theta = 1.0
sigma = .3

schwartz_one_factor = Schwartz1FactorModel(simuConfig, S0, kappa, theta, sigma)
schwartz_one_factor.simulate()
paths = schwartz_one_factor.Paths

#max_likelihood = MaxLikelihood(paths[0], schwartz_one_factor)
#success, kappa, theta = max_likelihood.calibrate()


#kappa = max_likelihood.calibrate_kappa()
#print(kappa)

#least_square = LeastSquare(paths[0], schwartz_one_factor)
#kappa, theta = least_square.calibrate()


theta_max_likelihood = np.zeros((nb_simus))
theta_least_square = np.zeros((nb_simus))
theta_max_likelihood_simul  = np.zeros((nb_simus))
kappa_max_likelihood = np.zeros((nb_simus))
kappa_least_square = np.zeros((nb_simus))
kappa_max_likelihood_simul = np.zeros((nb_simus))

for i in range(0, nb_simus):
    max_likelihood = MaxLikelihood(paths[i], schwartz_one_factor)
    least_square = LeastSquare(paths[i], schwartz_one_factor)
    success, kappa, theta = max_likelihood.calibrate()

    if not success:
        raise Exception('Simultaneous Calibration was not successful.')

    theta_max_likelihood[i], theta_least_square[i], theta_max_likelihood_simul = \
        max_likelihood.calibrate_theta(), least_square.calibrate()[0], theta
    kappa_max_likelihood[i], kappa_least_square[i], kappa_max_likelihood_simul = \
        max_likelihood.calibrate_kappa(), least_square.calibrate()[1], kappa

plt.plot(theta_max_likelihood)
plt.plot(theta_least_square)
plt.plot(theta_max_likelihood_simul)
plt.show()

plt.plot(kappa_max_likelihood)
plt.plot(kappa_least_square)
plt.plot(kappa_max_likelihood_simul)
plt.show()

print('Mean theta max likel.: {0} \n Mean theta lst. sq.: {1} \n Mean theta max likel. simul.: {2}'.format(np.mean(theta_max_likelihood),
    np.mean(theta_least_square), np.mean(theta_max_likelihood_simul)))

print('St. dev. theta max likel.: {0} \n St. dev. theta lst. sq.: {1} \n St. dev. theta max likel. simul.: {2}'.format(np.std(theta_max_likelihood),
    np.std(theta_least_square), np.std(theta_max_likelihood_simul)))

print('Mean kappa max likel.: {0} \n Mean kappa lst. sq.: {1} \n Mean kappa max likel. simul.: {2}'.format(np.mean(kappa_max_likelihood),
    np.mean(kappa_least_square), np.mean(kappa_max_likelihood_simul)))

print('St. dev. kappa max likel.: {0} \n St. dev. kappa lst. sq.: {1} \n St. dev. kappa max likel. simul.: {2}'.format(np.std(kappa_max_likelihood),
    np.std(kappa_least_square), np.std(kappa_max_likelihood_simul)))