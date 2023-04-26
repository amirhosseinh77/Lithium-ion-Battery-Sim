import numpy as np
import matplotlib.pyplot as plt
from elements.batteryModel import LithiumIonBattery, make_OCVfromSOCtemp, make_dOCVfromSOCtemp
from elements.barDelta_temp_all_params import BarDelta_SPKF
from scipy.linalg import block_diag, cholesky
from copy import copy

udds = np.loadtxt("models/udds.txt")
udds

ik = np.hstack([np.zeros(300), udds[:,1], np.zeros(300), udds[:,1], np.zeros(241)]*10)
time = np.arange(len(ik))/3600

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

# currents = np.zeros((len(t),1))
voltages = np.zeros((len(time),len(batteryPack)))
SOCs = np.zeros((len(time),len(batteryPack)))

for k,i in enumerate(ik):
    # currents[k] = i
    for c,cell in enumerate(batteryPack):
        newState = cell.stateEqn(i)
        voltage = cell.outputEqn(i)
        cell.updateState(newState)

        voltages[k,c] = voltage
        SOCs[k,c] = cell.z_k.ravel()


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.plot(time, voltages)
plt.grid()
plt.legend(['cell1','cell2','cell3','cell4'])
plt.xlabel('Time (hr)')
plt.ylabel('Voltage (V)')
plt.title('Voltage versus time for 4 cells')

plt.subplot(1,2,2)
plt.plot(time, SOCs)
plt.grid()
plt.legend(['cell1','cell2','cell3','cell4'])
plt.xlabel('Time (hr)')
plt.ylabel('State of charge (%)')
plt.title('SOC versus time for 4 cells')
plt.show()

##############################################################################

vk = voltages
zk = SOCs
T=25
z0, R0, Q0

ibias = 0.5
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


Ztrues = [soc[0]]
Zhats = []
Zbounds = []
ibhats = []
rhats = []
qinvhats = []

DZS = []
SCM1_SPKF = BarDelta_SPKF(batteryPack, SigmaX, SigmaW, SigmaV)


for i in range(len(current)):
    
    zhat, zbound, ibhat, rhat, qinvhat = SCM1_SPKF.iter_bar(current[i], voltage[i])
    SCM1_SPKF.iter_delta(current[i], voltage[i])
    # break
    # store data
    Ztrues.append(soc[i])
    Zhats.append(zhat)
    Zbounds.append(zbound)
    ibhats.append(ibhat)
    rhats.append(rhat)
    qinvhats.append(qinvhat)
    DZS.append(SCM1_SPKF.dz)


Ztrues = np.array(Ztrues)
Zhats = np.array(Zhats)
Zbounds = np.array(Zbounds)
ibhats = np.array(ibhats)
DZS = np.array(DZS)

print('very nice!')

plt.plot(time,DZS)
plt.show()

plt.plot(time,Zhats+DZS)
plt.plot(time,Ztrues[:-1])

plt.grid()
plt.title('Voltage')
# plt.xlabel('Time (min)')
plt.ylabel('Votltage (V)')
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(time,Zhats)
plt.subplot(2,1,2)
plt.plot(time,ibhats)

plt.show()

# # plot diagrams
# plt.figure()
# plt.subplot(2,2,1)
# plt.plot(time,current)
# plt.grid()
# plt.title('Current')
# # plt.xlabel('Time (min)')
# plt.ylabel('Current (A)')

# plt.subplot(2,2,2)
# plt.plot(time/60,voltage)
# plt.grid()
# plt.title('Voltage')
# # plt.xlabel('Time (min)')
# plt.ylabel('Votltage (V)')

# plt.subplot(2,2,3)
# plt.plot(time/60,100*Ztrues[:-1],color=(0,0.8,0))
# plt.plot(time/60,100*Zhats,color=(0,0,1),linestyle='dashed')
# plt.fill_between(time/60, 100*(Zhats+Zbounds), 100*(Zhats-Zbounds), alpha=0.3)
# plt.grid()
# plt.legend(['Truth','Estimate','Bounds'])
# plt.title('SOC estimation using SPKF')
# plt.xlabel('Time (min)')
# plt.ylabel('SOC (%)')

# plt.subplot(2,2,4)
# estErr = Ztrues[:-1]-Zhats 
# plt.plot(time/60, 100*estErr)
# plt.fill_between(time/60, 100*Zbounds, -100*Zbounds, alpha=0.3)
# plt.grid()
# plt.legend(['Estimation error','Bounds'])
# plt.title('SOC estimation errors using SPKF')
# plt.xlabel('Time (min)') 
# plt.ylabel('SOC error (%)')
# plt.show()
    