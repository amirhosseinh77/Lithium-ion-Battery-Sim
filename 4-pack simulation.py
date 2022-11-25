import numpy as np
import matplotlib.pyplot as plt
from elements.batteryModel import LithiumIonBattery, make_OCVfromSOCtemp, make_dOCVfromSOCtemp
from elements.barDelta import BarDelta_SPKF
from scipy.linalg import block_diag, cholesky
from copy import copy

udds = np.loadtxt("models/udds.txt")
udds

ik = np.hstack([np.zeros(300), udds[:,1], np.zeros(300), udds[:,1], np.zeros(241)])
t = np.arange(len(ik))/3600

# battery cells
z0 = np.arange(0.9, 0.6, -0.1)       # set initial SOC for each cell in pack
R0 = np.arange(1.3, 1, -0.1)*1e-3    # set R0 for each cell in pack
Q0 = np.arange(25, 29, 1)            # set Q for each cell in pack

print(z0, R0, Q0)

cell1 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=1)
cell2 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=1)
cell3 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=1)
cell4 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=1)

cell1.z_k, cell2.z_k, cell3.z_k, cell4.z_k = z0
cell1.R0Param, cell2.R0Param, cell3.R0Param, cell4.R0Param = R0
cell1.QParam, cell2.QParam, cell3.QParam, cell4.QParam = Q0

batteryPack = [cell1, cell2, cell3, cell4]

currents = np.zeros((len(t),1))
voltages = np.zeros((len(t),len(batteryPack)))
SOCs = np.zeros((len(t),len(batteryPack)))

for k,i in enumerate(ik):
    currents[k] = i
    
    for c,cell in enumerate(batteryPack):
        newState = cell.stateEqn(i)
        voltage = cell.outputEqn(i)
        cell.updateState(newState)

        voltages[k,c] = voltage
        SOCs[k,c] = cell.z_k.ravel()

'''
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.plot(t, voltages)
plt.grid()
plt.legend(['cell1','cell2','cell3','cell4'])
plt.xlabel('Time (hr)')
plt.ylabel('Voltage (V)')
plt.title('Voltage versus time for 4 cells')

plt.subplot(1,2,2)
plt.plot(t, SOCs)
plt.grid()
plt.legend(['cell1','cell2','cell3','cell4'])
plt.xlabel('Time (hr)')
plt.ylabel('State of charge (%)')
plt.title('SOC versus time for 4 cells')
plt.show()
'''
##############################################################################

vk = voltages
ik = currents
zk = SOCs
T=25
z0, R0, Q0

ibias = 0.5
time = np.arange(len(ik))
deltat = 1

current = ik + ibias
voltage = vk
soc = zk

# Reserve storage for computed results, for plotting
sochat = 0*time    # reserve storage for bar-soc values
socbound = sochat  # and bounds on those values
bias = sochat      # ... also for current-sensor bias estimate
biasBound = sochat # and bounds on that estimate

dsochat = np.zeros_like(voltage)   # reserve storage for delta-soc values
dsocbound = np.zeros_like(voltage) # and for bounds on those values
dR0 = np.zeros_like(voltage)        # reserve storage for delta-R0 values
dR0bound = np.zeros_like(voltage)   # and for bounds on those values

# Covariance values
# State ordering: ir,h,z,bias,R,Qinv
SigmaX = block_diag(1e2, 1e-4, 1e-2, 5e-2, 5e-2, 5e-2)  # uncertainty of initial state (ir,h,z,bias,R0,Qinv)
SigmaW = block_diag(1e-1, 1e-4, 1e-4, 1e-4)             # uncertainty of current sensor, bias, R0, Qinv
SigmaV = block_diag(1e-3)                               # uncertainty of voltage sensor, output equation

SCM1_SPKF = BarDelta_SPKF(batteryPack, SigmaX, SigmaW, SigmaV)
SCM1_SPKF.iter_bar(current[0], voltage[0])
SCM1_SPKF.iter_delta(current[0], voltage[0])

print('every thing gooooooood!!!')