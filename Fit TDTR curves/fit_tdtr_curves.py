import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoMinorLocator

PATH_TDTR = "example_tdtr_file.txt"
FIT_TYPE = "full_exp"
SHIFT = 0
TALE_CUT = 0
PUMP_DURATION = 20 # [us]


def cut_and_normalize(x, y):
    '''Cut the increasing part of the signal and normalize the rest'''
    N = len(y)
    cut = find_nearest(x, PUMP_DURATION*1e-6) + SHIFT
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
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'
    # plt.plot(x-PUMP_DURATION, y, color='r')
    fig, (ax1) = plt.subplots(1, 1, figsize = (5, 3.5), dpi = 100)

    # PLOT THE CURVES
    ax1.plot(x_norm*1e6, y_norm, color='#1c68ff')
    ax1.plot(x_norm*1e6, a*np.exp(-x_norm/t)+d, color='k', linewidth=1.5)  
    # LABELS
    ax1.set_ylabel('Normalized probe signal', fontsize=14)
    ax1.set_xlabel('Time (μs)', fontsize=14)
    ax1.legend(['Experiment', 'Fit'], framealpha = 0.0, loc = 'upper right')

    # RANGE
    # ax1.set_xlim([-2e-6,24e-6])
    ax1.set_ylim([-0.1,1.1])

    # AXIS STYLING
    ax1.tick_params(direction='in', which='major', length=6)
    ax1.tick_params(direction='in', which='minor', length=3)
    ax1.minorticks_on()
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.tight_layout()

    fig.savefig('Figure_TDTR_curve_with_fit.pdf')
    plt.show()
    print ("Decay time = ", t*1e6, "μs")
    print ("Amplitude = ", a)
    print ("Offset = ", d)


def main():
    x, y  = np.genfromtxt(PATH_TDTR, unpack = True,  delimiter='\t', usecols = (0,2), skip_header = 0)
    x_norm, y_norm = cut_and_normalize(x, y)
    t, d, a = exp_fit(x_norm, y_norm, FIT_TYPE)
    plot(x, y, x_norm, y_norm, a, d, t)


if __name__ == "__main__":
    main()
