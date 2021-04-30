import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PATH_TDTR = "example_tdtr_file.txt"
FIT_TYPE = "full_exp"
SHIFT = 0
TALE_CUT = 0
PUMP_DURATION = 20e-6


def cut_and_normalize(x, y):
    '''Cut the increasing part of the signal and normalize the rest'''
    N = len(y)
    cut = find_nearest(x, PUMP_DURATION) + SHIFT
    x_norm = x[range(cut,N-TALE_CUT)] - x[cut]
    y_norm = y[range(cut,N-TALE_CUT)] / y[cut]
    return x_norm, y_norm


def find_nearest(array, value):
    '''Find nearest value in the array'''
    index = (np.abs(array - value)).argmin()
    return index


def exp_fit(x_norm, y_norm, FIT_TYPE):
    '''This function outputs fit parameters for various exponential fits '''    
    if FIT_TYPE == 'full_exp':
        def full_exp(x, t, d, a):
            return a * np.exp(-x/t) + d
        par, pcov = curve_fit(full_exp, x_norm, y_norm, bounds=(0, [100e-5, 0.05, 1.0]))
        t, d, a = par
    if FIT_TYPE == 'exp':
        def exp(x, t, d):
            return np.exp(-x/t) + d
        par, pcov = curve_fit(exp, x_norm, y_norm, bounds=(0, [100e-5, 0.05]))
        t, d = par
        a = 1.0
    return t, d, a


def plot(x, y, x_norm, y_norm, a, d, t):
    '''This function plots the graph'''
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = "10"
    # plt.plot(x-PUMP_DURATION, y, color='r')
    plt.plot(x_norm, y_norm, color='#1c68ff')
    plt.plot(x_norm, a*np.exp(-x_norm/t)+d, color='k', linewidth=1.5)  
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized TDTR signal')
    plt.show()
    print ("Decay time = ", t*1e6, "us")
    print ("Amplitude = ", a)
    print ("Offset = ", d)
    return


def main():
    x, y  = np.genfromtxt(PATH_TDTR, unpack = True,  delimiter='\t', usecols = (0,2), skip_header = 0)
    x_norm, y_norm = cut_and_normalize(x, y)
    t, d, a = exp_fit(x_norm, y_norm, FIT_TYPE)
    plot(x, y, x_norm, y_norm, a, d, t)
    return

main()
