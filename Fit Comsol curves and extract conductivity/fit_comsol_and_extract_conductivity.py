import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PATH_TO_COMSOL_FILE = "example_comsol_file.txt"

MEASURED_DECAY_TIMES = np.array([1.5, 1.55, 1.6])  # [us]
# PATH_TO_DECAY_TIMES_FILE = "DT_300K.csv"

PUMP_DURATION = 1 # [us]

COLUMN_OF_RADII       = 0
COLUMN_OF_TIME        = 1
COLUMN_OF_KAPPA       = 2
COLUMN_OF_TEMPERATURE = 3

FIT_TYPE = 'full_exp'
SHIFT    = 0
TALE_CUT = 0


def read_decay_time_file(path):
    '''Here we read the file of decay times if it is provided'''
    data = np.genfromtxt(path, delimiter=',', usecols = (0,1,2), skip_header = 1)
    return data


def read_file(path):
    '''Reading file form Comsol, separating into columns etc'''
    original_data = np.loadtxt(path, comments='%')
                                                                                   
    number_of_radiuses = len(np.unique(original_data[:,COLUMN_OF_RADII]))
    number_of_kappa = len(np.unique(original_data[:,COLUMN_OF_KAPPA])) 
    number_of_nodes = len(original_data[:,0])//(number_of_kappa*number_of_radiuses)
    kappas = np.unique(original_data[:,COLUMN_OF_KAPPA])

    data = np.zeros((number_of_nodes, number_of_kappa+1))                                                  
    data[:,0] = original_data[range(number_of_nodes), COLUMN_OF_TIME]
    for j in range(number_of_kappa):
        data[:,j+1] = original_data[range(j*(number_of_nodes),(j+1)*(number_of_nodes)), COLUMN_OF_TEMPERATURE]
    return data, kappas


def cut_and_normalize(x, y):
    '''Cutting the increasing part of the signal an normalizing the rest between 0 and 1'''
    cut = find_nearest(x, PUMP_DURATION*1e-6) + SHIFT
    x_norm = x[range(cut, len(y) - TALE_CUT)] - x[cut]
    y -= min(y) 
    y_norm = y[range(cut, len(y) - TALE_CUT)] / y[cut] 
    return x_norm, y_norm


def find_nearest(array, value):
    '''Finding the nearest value in the array and returning its number'''
    index = (np.abs(array - value)).argmin()
    return index


def exp_fit(x_norm, y_norm):
    '''This function outputs fit parameters for various exponential fits '''    
    if FIT_TYPE == 'full_exp':
        def full_exp(x, t, d, a):
            return a * np.exp(-x/t) + d
        par, _ = curve_fit(full_exp, x_norm, y_norm, bounds=(0, [100e-5, 0.1, 1.1]))
        t, d, a = par

    if FIT_TYPE == 'exp':
        def exp(x, t, d):
            return np.exp(-x/t) + d
        par, _ = curve_fit(exp, x_norm, y_norm, bounds=(0, [100e-5, 0.1]))
        t, d = par
        a = 1.0
    return t, d, a


def plot_decay_curves(x, y, a, d, t):
    '''This function plots the graph'''
    # COLORS = ['#1c68ff','#000000','#FB0071','#00FB76']
    plt.plot(x*1e6, y, 'o', mfc='none', color='#1c68ff', linewidth=1.0, markersize=3)
    plt.plot(x*1e6, a*np.exp(-x/t) + d, color='k', linewidth=1.5)  
    plt.ylabel('Normalized probe signal', fontsize=14)
    plt.xlabel('Time (μs)', fontsize=14)


def extracting_thermal_conductivity(decay_times, kappas, measured_decay_times):
    '''Here we extract the thermal_conductivities from the obtained dependence of 
    decay times on the thermal conductivities in Comsol'''
    A, B, C = np.polyfit(decay_times, kappas, 2)
    polynome = lambda t: A*t**2 + B*t + C
    fit_curve_x = np.linspace(decay_times[0], decay_times[-1], num=30)
    fit_curve_y = np.array([polynome(t) for t in fit_curve_x])
    thermal_conductivities = np.array([polynome(t) for t in measured_decay_times]) 
    for index, kappa in enumerate(kappas):
        plt.plot(decay_times[index]*1e6, kappa, 'o', mfc='none', color='#1c68ff', linewidth=1.0)
    plt.plot(fit_curve_x*1e6, fit_curve_y, color='k', linewidth=1.5) 
    for time, conductivity in zip(measured_decay_times, thermal_conductivities):
        plt.scatter(time*1e6, conductivity, color='k', s=60) 
    plt.xlabel('Decay time (μs)', fontsize=14)
    plt.ylabel('Thermal conductivity (W/mK)', fontsize=14) 
    plt.savefig('Figure_Thermal_conductivity.pdf')
    plt.show()
    return thermal_conductivities


def main():
    original_data, kappas = read_file(PATH_TO_COMSOL_FILE)
    decay_times = []
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'
    for i in range(original_data.shape[1]-1):
        x, y = cut_and_normalize(original_data[:,0], original_data[:,i+1])
        t, d, a = exp_fit(x, y)
        decay_times.append(t)
        plot_decay_curves(x, y, a, d, t)
    plt.savefig('Figure_Comsol_fits.pdf')
    plt.show()
    
    if np.size(MEASURED_DECAY_TIMES) > 0:
        measured_decay_times = MEASURED_DECAY_TIMES
    else:
        measured_decay_times = read_decay_time_file(PATH_TO_DECAY_TIMES_FILE)
    measured_decay_times *= 1e-6 # Convert to seconds

    thermal_conductivities = extracting_thermal_conductivity(decay_times, kappas, measured_decay_times)

    np.savetxt("Thermal conductivies.csv", thermal_conductivities, delimiter=",", fmt='%1.4f', header="K(W/mK)")
    print ("Thermal conductivity = ", thermal_conductivities, "(W/mK)")


if __name__ == '__main__':
    main()
