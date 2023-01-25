
import numpy as np
from scipy import optimize
from scipy.optimize import Bounds
from scipy import stats


def lognormparams(confidence, interval):
    # guess_mean = (interval[0] + interval[1]) / 2.0
    guess_mean = 0.000001
    guess_std = np.sqrt(0.5 * ((interval[0] - guess_mean) ** 2 + (interval[1] - guess_mean) ** 2)) * confidence

    print('Guess Mean ', guess_mean, ' Guess STD', guess_std)

    initial_guess = np.array([guess_mean, guess_std])

    log_bounds = Bounds(lb=np.array([0, 0]))
    opt_sol = optimize.minimize(LogNIntvErr, initial_guess,
                                args=(confidence, interval), method='Nelder-Mead', bounds=log_bounds)

    return opt_sol['x']


def GammaParamsSearch(conf, interv):
    init_alpha = 1.1
    guess_mean = (interv[0] + interv[1]) / 2.0
    init_beta = init_alpha / guess_mean
    init_guess = np.array([init_alpha, init_beta])

    gamma_bound = Bounds(lb=[1 + 1e-6, 0])
    opt_sol = optimize.minimize(GammaIntErr, init_guess,
                                args=(conf, interv), method='Nelder-Mead', bounds=gamma_bound)

    return opt_sol['x']

def NormParamsSearch(conf, interv):
    guess_mean = (interv[0] + interv[1]) / 2.0
    guess_std = np.sqrt((interv[0] - guess_mean)**2 + (interv[1] - guess_mean)**2)
    init_guess = np.array([guess_mean, guess_std])
    norm_bound = Bounds(lb =[interv[0], 0], ub=[interv[1], np.infty])
    opt_sol = optimize.minimize(NormIntErr, init_guess, args=(conf, interv), method="Nelder-Mead", bounds=norm_bound)
    return opt_sol['x']


def LogNIntvErr(guess, *args):
    conf, actual = args
    guess_std = guess[1]
    guess_mean = guess[0]
    guess_int = stats.lognorm.interval(conf, guess_std, guess_mean)
    print(guess_int)
    return (guess_int[0] - actual[0]) ** 2 + (guess_int[1] - actual[1]) ** 2


def GammaIntErr(guess, *args):
    conf, actual = args
    guess_beta = guess[1]
    guess_alpha = guess[0]
    guess_int = stats.gamma.interval(conf, a=guess_alpha, loc=0, scale=guess_beta)
    return (guess_int[0] - actual[0]) ** 2 + (guess_int[1] - actual[1]) ** 2

def NormIntErr(guess, *args):
    conf, actual = args
    guess_mean = guess[0]
    guess_std = guess[1]
    guess_interv = stats.norm.interval(conf, loc=guess_mean, scale=guess_std)
    return (guess_interv[0] - actual[0])**2 + (guess_interv[1] - actual[1])**2
