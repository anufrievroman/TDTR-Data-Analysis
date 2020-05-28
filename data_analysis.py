# This code is written by Roman Anufriev for Nomura lab, IIS, University of Tokyo in 2018.
# The code analyses decay curves produced by Comsol cimulation, fits them and extracts thermal conductivity using experimentally measured decay time.
# Contact me by anufriev.roman@protonmail.com if you have any questions.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename="test_file.txt"   # Input file from Comsol
measured_decay_time=1.6e-6 # [s]

# DEFINE HERE WHICH COLUMN MEANS WHAT IN YOUR INPUT FILE
# Remember that in python we count from zero, so the first column is 0, not 1
column_of_radii=0
column_of_time=1
column_of_kappa=2
column_of_temperature=3

fit_type='full_exp' # or 'log' or 'exp'                                         # There are three types of fitting, 'exp','log', or 'full_exp' 

shift=0                                                                         # Don't change unless you know what you're doing

# TAKING CARE OF THE INPUT FILE AND ITS DATA
with open(filename, "r") as f:
    original_data = np.loadtxt(f, comments='%')
                                                                               
number_of_radiuses=len(np.unique(original_data[:,column_of_radii]))
number_of_kappa=len(np.unique(original_data[:,column_of_kappa])) 
number_of_nodes=len(original_data[:,0])//(number_of_kappa*number_of_radiuses)
kappas=np.unique(original_data[:,column_of_kappa])

data=np.zeros((number_of_nodes+0,number_of_kappa+1))                                                  
data[:,0]=original_data[range(number_of_nodes),column_of_time]                  # The first column is time - the same for all kappas and radiuses

for j in range(number_of_kappa):                                                # Let's separete the data for different kappa into different columns
    data[:,j+1]=original_data[range(j*(number_of_nodes),(j+1)*(number_of_nodes)),column_of_temperature]

for j in range(number_of_kappa):                                              
    data[:,j+1]=data[:,j+1]-min(data[:,j+1])                                    # Let's bring base line to zero 
    data[:,j+1]=data[:,j+1]/max(data[:,j+1])                                    # Let's normalise, so now it is from zero to one

for i in range(number_of_nodes):                                                # Let's cut off the part before the decay
    if data[i,1]==1:  cutoff_node_number=i+shift				# Here we found where is 1, so it must be the pick

data_normalized=np.zeros((number_of_nodes-cutoff_node_number,number_of_kappa+1))
data_normalized[:,0]=data[range(cutoff_node_number,number_of_nodes),0]-data[cutoff_node_number,0]

for j in range(number_of_kappa):                                                # And here we will keep only the decaying part of the curve
    data_normalized[:,j+1]=data[range(cutoff_node_number,number_of_nodes),j+1]

# TYPES OF FITTING
def simple_exp_fit(np_array_x, np_array_y):
    '''This function outputs parameter t of the exponential fit y = exp[-x/t]'''   
    def func(x, t):
        return np.exp(-x/t)
    par, pcov = curve_fit(func, np_array_x, np_array_y) 
    return par

def simple_exp_fit_via_log(np_array_x, np_array_y):
    '''This function outputs parameter t of the exponential fit y = exp[-x/t]'''   
    def func(x, a):
        return a*x
    par, pcov = curve_fit(func, np_array_x, np.log(np_array_y) )#, bounds=(0, [3., 100., 400.,1.])) 
    return 1/abs(par)

def full_exp_fit(np_array_x, np_array_y):
    '''This function outputs parameter t of the exponential fit y = a*exp[-x/t]+d'''    
    def func(x, a, t, d):
        return a * np.exp(-x/t) + d
    #make the curve_fit IMPORTANT Control the parameters bounds here!
    par, pcov = curve_fit(func, np_array_x, np_array_y, bounds=(0, [1.1, 100e-5, 1e-5]))
    return par

# FITTING THE DECAY CURVES 
if fit_type=='exp':
    decay_times=[float(simple_exp_fit(data_normalized[:,0],data_normalized[:,j+1])) for j in range(number_of_kappa)] 
elif fit_type=='log':
    decay_times=[simple_exp_fit_via_log(data_normalized[:,0],data_normalized[:,j+1]) for j in range(number_of_kappa)] 
elif fit_type=='full_exp':
    #decay_times=[full_exp_fit(data_normalized[:,0],data_normalized[:,j+1]) for j in range(number_of_kappa)] 
    decay_times=[0]*number_of_kappa
    a_coefficients=[0]*number_of_kappa
    d_coefficients=[0]*number_of_kappa
    for j in range(number_of_kappa):
        fit_parameters = full_exp_fit(data_normalized[:,0],data_normalized[:,j+1])
        decay_times[j] = fit_parameters[1]
        a_coefficients[j] = fit_parameters[0]
        d_coefficients[j] = fit_parameters[2]

# PLOTING THE RESULTS OF FITTING  
for j in range(0,number_of_kappa):
    plt.plot(data_normalized[:,0]*1e6,data_normalized[:,j+1], 'o', mfc='none',color='b', linewidth=1.0, markersize=2)
    
if fit_type=='exp' or fit_type=='log':
    for j in range(0,number_of_kappa):
        plt.plot(data_normalized[:,0]*1e6,np.exp(-data_normalized[:,0]/decay_times[j]), color='k', linewidth=1.5)  
elif fit_type=='full_exp':
    for j in range(0,number_of_kappa):
        plt.plot(data_normalized[:,0]*1e6,a_coefficients[j]*np.exp(-data_normalized[:,0]/decay_times[j])+d_coefficients[j], color='k', linewidth=1.5)  
        
plt.xlabel('Decay time (us)', fontsize=12)
plt.ylabel('Normalized TDTR signal', fontsize=12)
plt.savefig("Decay curve fitting.pdf", dpi=300, format = 'pdf', bbox_inches="tight")
plt.show()

# FITTING THE DEPENDENCE OF DECAY TIMES ON THERMAL CONDUCTIVITY
coefs = np.polyfit(decay_times, kappas, 2)
kappa_fit_curve=np.zeros((100,2))
kappa_fit_curve[:,0]=[decay_times[number_of_kappa-1]+i*(decay_times[0]-decay_times[number_of_kappa-1])/100 for i in range(100)]
kappa_fit_curve[:,1]=coefs[0]*kappa_fit_curve[:,0]**2+coefs[1]*kappa_fit_curve[:,0]+coefs[2]

thermal_conductivity=coefs[0]*measured_decay_time**2+coefs[1]*measured_decay_time+coefs[2]

for j in range(0,number_of_kappa):
    plt.plot(decay_times[j]*1e6,kappas[j], 'o', mfc='none',color='b', linewidth=1.0)
plt.plot(kappa_fit_curve[:,0]*1e6,kappa_fit_curve[:,1], color='k', linewidth=1.5) 
plt.scatter(measured_decay_time*1e6,thermal_conductivity, color='k', s=100) 
plt.xlabel('Decay time (us)', fontsize=12)
plt.ylabel('Thermal conductivity (W/mK)', fontsize=12) 
plt.show()

print ('\nThermal conductivity =', thermal_conductivity, '(W/mK)')
