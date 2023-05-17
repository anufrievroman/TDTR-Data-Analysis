import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PATH_TO_COMSOL_FILE = "example_comsol_file.txt"
MEASURED_DECAY_TIMES = np.array([1.6])  # [us]
SWEEP_PARAMETER_NUMBER = 0
PUMP_DURATION          = 1 # [us]

COLUMN_OF_SWEEP_PARAMETER = 0
COLUMN_OF_TIME            = 1
COLUMN_OF_KAPPA           = 2
COLUMN_OF_TEMPERATURE     = 3

FIT_TYPE       = 'full_exp'
SHIFT          = 0
TALE_CUT       = 0
SAVEFIGURES    = False
SAVEDATA       = False
POLYNOMEDEGREE = 4


def read_comsol_file():
    '''Reading file form Comsol, separating into columns according to the kappa'''
    original_data = np.loadtxt(PATH_TO_COMSOL_FILE, comments='%')
    kappas, counts = np.unique(original_data[:,COLUMN_OF_KAPPA], return_counts=True)
    data = []
    for kappa in kappas:
        # Create a boolean mask to filter rows with the current value:
        mask = original_data[:, COLUMN_OF_KAPPA] == kappa
        splitted_part = original_data[mask][:, [COLUMN_OF_TIME, COLUMN_OF_TEMPERATURE]]
        data.append(splitted_part)
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
    return (np.abs(array - value)).argmin()


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


def extracting_thermal_conductivity(decay_times, kappas, measured_decay_times):
    '''Here we extract the thermal_conductivities from the obtained dependence of
    decay times on the thermal conductivities in Comsol'''
    if POLYNOMEDEGREE == 2:
        A, B, C = np.polyfit(decay_times, kappas, 2)
        polynome = lambda t: A*t**2 + B*t + C
    elif POLYNOMEDEGREE == 3:
        A, B, C, D = np.polyfit(decay_times, kappas, 3)
        polynome = lambda t: A*t**3 + B*t**2 + C*t + D
    elif POLYNOMEDEGREE == 4:
        A, B, C, D, E = np.polyfit(decay_times, kappas, 4)
        polynome = lambda t: A*t**4 + B*t**3 + C*t**2 + D*t + E
    fit_curve_x = np.linspace(decay_times[0], decay_times[-1], num=30)
    fit_curve_y = np.array([polynome(t) for t in fit_curve_x])
    thermal_conductivities = np.array([polynome(t) for t in measured_decay_times])

    fig2, ax2 = plt.subplots(1, 1)
    for index, kappa in enumerate(kappas):
        ax2.plot(decay_times[index]*1e6, kappa, 'o', mfc='none', color='#1c68ff', linewidth=1.0)
    plt.plot(fit_curve_x*1e6, fit_curve_y, color='k', linewidth=1.5)
    for time, conductivity in zip(measured_decay_times, thermal_conductivities):
        ax2.scatter(time*1e6, conductivity, color='k', s=60)
    ax2.set_xlabel('Decay time (μs)', fontsize=14)
    ax2.set_ylabel('Thermal conductivity (W/mK)', fontsize=14)
    if SAVEFIGURES:
        fig2.savefig('Figure_Thermal_conductivity.pdf')
    plt.show()
    return thermal_conductivities


def main():
    data, kappas = read_comsol_file()
    decay_times = []
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'

    fig1, ax1 = plt.subplots(1, 1)
    for i, _ in enumerate(kappas):
        x, y = cut_and_normalize(data[i][:,0], data[i][:,1])
        t, d, a = exp_fit(x, y)
        decay_times.append(t)
        ax1.plot(x*1e6, y, 'o', mfc='none', color='#1c68ff', linewidth=1.0, markersize=3)
        ax1.plot(x*1e6, a*np.exp(-x/t) + d, color='k', linewidth=1.5)
        ax1.set_ylabel('Normalized probe signal', fontsize=14)
        ax1.set_xlabel('Time (μs)', fontsize=14)

        if SAVEDATA:
            output_data = np.vstack((x*1e6, y, a*np.exp(-x/t) + d)).T
            np.savetxt(f"Fit{i}.csv", output_data, delimiter=",", fmt='%1.5f', header="t(us),Comsol,Fit")

    if SAVEFIGURES:
        fig1.savefig('Figure_Comsol_fits.pdf')
    plt.show()

    measured_decay_times = MEASURED_DECAY_TIMES*1e-6 # Convert to seconds
    thermal_conductivities = extracting_thermal_conductivity(decay_times, kappas, measured_decay_times)

    if SAVEDATA:
        output_data = np.vstack((decay_times, kappas)).T
        np.savetxt(f"Decay_times.csv", output_data, delimiter=",", fmt='%1.9f', header="t(s),K(W/mK)")
        np.savetxt("Thermal conductivies.csv", thermal_conductivities, delimiter=",", fmt='%1.9f', header="K(W/mK)")
    print ("Thermal conductivity = ", thermal_conductivities, "(W/mK)")


if __name__ == '__main__':
    main()
