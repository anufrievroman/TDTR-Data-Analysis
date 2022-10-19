import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# INPUT FILES
PATH_TDTR = ["example_tdtr_file2.txt",]      # It can be more than one file
PATH_COMSOL = "example_comsol_file2.txt"

COLORS = ['#000000','#363e5c','#1c68ff','#FB0071','#00FB76','#000000','#1c68ff','#FB0071','#00FB76']

RENORMALIZATION_CONSTANT = 1.08
TDTR2 = False

# READING THE EXPERIMENTAL DATA
times=[]
signals=[]
for i in range(len(PATH_TDTR)):
    delim = ',' if TDTR2 else '\t'
    t, R = np.genfromtxt(PATH_TDTR[i], unpack = True,  delimiter=delim, usecols = (0, 2), skip_header = 0)
    R *= RENORMALIZATION_CONSTANT
    times.append(t)
    signals.append(R)


# READING THE COMSOL DATA
column_of_radii       = 0
column_of_time        = 1
column_of_kappa       = 2
column_of_temperature = 3

original_data = np.loadtxt(PATH_COMSOL, comments='%')


# TAKING CARE OF THE COMSOL DATA
number_of_radiuses=len(np.unique(original_data[:,column_of_radii]))
number_of_kappa=len(np.unique(original_data[:,column_of_kappa])) 
number_of_nodes=len(original_data[:,0])//(number_of_kappa*number_of_radiuses)
kappas=np.unique(original_data[:,column_of_kappa])

data=np.zeros((number_of_nodes+0, number_of_kappa+1))
data[:,0]=original_data[range(number_of_nodes),column_of_time]          # The first column is time - the same for all kappas and radiuses

for j in range(number_of_kappa):                                        # Let's separete the data for different kappa into different columns
    data[:,j+1]=original_data[range(j*(number_of_nodes),(j+1)*(number_of_nodes)),column_of_temperature]

for j in range(number_of_kappa):
    data[:,j+1]=data[:,j+1]-min(data[:,j+1])                            # Let's bring base line to zero 
    data[:,j+1]=data[:,j+1]/max(data[:,j+1])                            # Let's normalize, so now it is from zero to one


# PLOTING THE DATA TOGETHER
plt.rcParams['font.family'] = "Arial"
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='6'
fig, (ax1) = plt.subplots(1, 1, figsize = (5, 3.5), dpi = 100)

for i in range(len(PATH_TDTR)):
    ax1.plot(times[i]*1e6-0.5, signals[i], '-', color='#3b8f60', linewidth=0.7)

for j in range(0,number_of_kappa):
    plt.plot(data[:,0]*1e6,data[:,j+1], '-', mfc='none', color=COLORS[j], linewidth=1.0, markersize=2)

# LABELS
ax1.set_ylabel('Probe signal', fontsize=14)
ax1.set_xlabel('Time (μs)', fontsize=14)
ax1.legend(['Exp.', kappas[0], kappas[1],kappas[2], kappas[3]], title='$\kappa_{SL}$ (Wm$^{-1}$K$^{-1}$)', framealpha = 0.0, loc = 'upper right')

# RANGE
ax1.set_xlim([-2,45])
ax1.set_ylim([-0.1,1.1])

# AXIS STYLING
ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
ax1.tick_params(direction='in', which='major', length=6)
ax1.tick_params(direction='in', which='minor', length=3)
ax1.minorticks_on()
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
plt.tight_layout()

fig.savefig('Figure_simulation_and_experiment.pdf')
plt.show()
