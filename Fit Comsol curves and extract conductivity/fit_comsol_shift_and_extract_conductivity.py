import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PATH_TO_COMSOL_FILE = "Comsol data/AuNW_L20000nm_W500nm.txt"
PATH_TO_TDTR_FILE = "Data/W1L4B.csv"

TDTR_ZERO_SHIFT = - 390  # μs
SWEEP_PARAMETER_NUMBER = 0

COLUMN_OF_SWEEP_PARAMETER = 0
COLUMN_OF_TIME            = 1
COLUMN_OF_KAPPA           = 2
COLUMN_OF_TEMPERATURE     = 3
FIT_TYPE       = 'full_exp'
SAVEFIGURES    = False
SAVEDATA       = False
POLYNOMEDEGREE_1 = 4
POLYNOMEDEGREE_2 = 3
FIT_RANGE_SIZE = 40

def read_comsol_file(path):
    '''Reading file form Comsol, separating into columns etc'''
    original_data = np.loadtxt(path, comments='%')
    number_of_radiuses = len(np.unique(original_data[:,COLUMN_OF_SWEEP_PARAMETER]))
    number_of_kappa = len(np.unique(original_data[:,COLUMN_OF_KAPPA]))
    number_of_nodes = len(original_data[:,0])//(number_of_kappa*number_of_radiuses)
    kappas = np.unique(original_data[:,COLUMN_OF_KAPPA])

    data = np.zeros((number_of_nodes, number_of_kappa+1))
    offset = number_of_nodes*number_of_kappa*SWEEP_PARAMETER_NUMBER
    data[:,0] = original_data[range(number_of_nodes), COLUMN_OF_TIME]
    for j in range(number_of_kappa):
        data[:,j+1] = original_data[range(offset + j*(number_of_nodes), offset + (j+1)*(number_of_nodes)), COLUMN_OF_TEMPERATURE]
    return data, kappas


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

def extract_thermal_conductivity(comsol_peaks, comsol_kappas, tdtr_peak):
    '''Extract the thermal_conductivities from the obtained dependence of
    peaks on the thermal conductivities in Comsol'''
    if POLYNOMEDEGREE_2 == 2:
        A, B, C = np.polyfit(comsol_peaks, comsol_kappas, 2)
        polynome = lambda t: A*t**2 + B*t + C
    elif POLYNOMEDEGREE_2 == 3:
        A, B, C, D = np.polyfit(comsol_peaks, comsol_kappas, 3)
        polynome = lambda t: A*t**3 + B*t**2 + C*t + D
    elif POLYNOMEDEGREE_2 == 4:
        A, B, C, D, E = np.polyfit(comsol_peaks, comsol_kappas, 4)
        polynome = lambda t: A*t**4 + B*t**3 + C*t**2 + D*t + E
    fit_curve_x = np.linspace(comsol_peaks[0], comsol_peaks[-1], num=30)
    fit_curve_y = np.array([polynome(t) for t in fit_curve_x])
    thermal_conductivity = polynome(tdtr_peak)

    fig2, ax2 = plt.subplots(1, 1)
    for peak, kappa in zip(comsol_peaks, comsol_kappas):
        ax2.plot(peak*1e6, kappa, 'o', mfc='none', color='#1c68ff', linewidth=1.0)
    plt.plot(fit_curve_x*1e6, fit_curve_y, color='k', linewidth=1.5)
    ax2.scatter(tdtr_peak*1e6, thermal_conductivity, color='k', s=60)
    ax2.set_xlabel('Delay time (μs)', fontsize=14)
    ax2.set_ylabel('Thermal conductivity (W/mK)', fontsize=14)
    if SAVEFIGURES:
        fig2.savefig('Figure_Thermal_conductivity.pdf')
    plt.show()
    return thermal_conductivity


def main():

    # PROCESS COMSOL DATA:
    comsol_data, comsol_kappas = read_comsol_file(PATH_TO_COMSOL_FILE)

    comsol_peaks = []
    fig1, ax1 = plt.subplots(1, 1)
    for i in range(comsol_data.shape[1]-1):
        x = comsol_data[:,0]
        y = comsol_data[:, i+1]
        y = (y - min(y))/max((y - min(y)))

        # Find a peak value:
        peak_x = x[find_nearest(y, np.max(y))]
        comsol_peaks.append(peak_x)

        ax1.plot(x*1e6, y, '-', mfc='none', linewidth=1.0, markersize=3)
        ax1.axvline(peak_x*1e6, linestyle='--', label='Peak')
        ax1.set_ylabel('Normalized probe signal', fontsize=14)
        ax1.set_xlabel('Time (μs)', fontsize=14)

    if SAVEFIGURES:
        fig1.savefig('Figure_Comsol_fits.pdf')
    plt.show()

    # PROCESS TDTR DATA:

    tdtr_data = np.loadtxt(PATH_TO_TDTR_FILE, delimiter=',')
    x = tdtr_data[:,0] + TDTR_ZERO_SHIFT*1e-6
    y = tdtr_data[:,1]

    # Define a small range around the minimum value:
    min_index = np.argmin(y)
    range_size = FIT_RANGE_SIZE
    range_start = max(0, min_index - range_size)
    range_end = min(len(x) - 1, min_index + range_size)
    x_range = x[range_start:range_end+1]
    y_range = y[range_start:range_end+1]

    # Fit polynomial to data within the range:
    poly_coeffs = np.polyfit(x_range, y_range, POLYNOMEDEGREE_1)
    poly_fit = np.poly1d(poly_coeffs)

    # Determine the minimum value of the polynomial within the range:
    poly_range = np.linspace(x_range[0], x_range[-1], 1000)
    tdtr_peak = poly_range[np.argmin(poly_fit(poly_range))]

    # Plot data and polynomial fit within the range:
    plt.plot(x_range*1e6, y_range, 'o',  color='#000000', label='Data')
    plt.plot(poly_range*1e6, poly_fit(poly_range), color='#1c68ff', label='Polynomial fit')
    plt.axvline(tdtr_peak*1e6, color='#D80382', linestyle='--', label='Minimum')
    plt.legend()
    plt.show()

    # EXTRACT THE THERMAL CONDUCTIVITY:
    thermal_conductivity = extract_thermal_conductivity(comsol_peaks, comsol_kappas, tdtr_peak)

    # if SAVEDATA:
        # output_data = np.vstack((decay_times, kappas)).T
        # np.savetxt(f"Decay_times.csv", output_data, delimiter=",", fmt='%1.9f', header="t(s),K(W/mK)")
        # np.savetxt("Thermal conductivies.csv", thermal_conductivities, delimiter=",", fmt='%1.9f', header="K(W/mK)")
    print ("Thermal conductivity = ", thermal_conductivity, "(W/mK)")


if __name__ == '__main__':
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'
    main()
