from elements.batteryModel import LithiumIonBattery, make_OCVfromSOCtemp, make_dOCVfromSOCtemp
from elements.estimator import SPKF
from elements.plots import plot_SOC
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from scipy.linalg import block_diag

data = loadmat('models/PANdata_P25.mat')
time    = np.array(data['DYNData']['script1']['time'])  
deltat = time[1]-time[0]
time    = time-time[0] # start time at 0
current = np.array(data['DYNData']['script1']['current']) # discharge > 0; charge < 0.
voltage = np.array(data['DYNData']['script1']['voltage'])
soc     = np.array(data['DYNData']['script1']['soc'])

############################## Simulate Model ##############################
LIB1 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=deltat)
        
SigmaX = block_diag(1e2,1e-2,1e-3)  # uncertainty of initial state
SigmaV = block_diag(3e-1)  # Uncertainty of voltage sensor, output equation
SigmaW = block_diag(4e0)   # Uncertainty of current sensor, state equation
LIB1_SPKF = SPKF(LIB1, SigmaX, SigmaV, SigmaW)

Ztrues = [soc[0]]
Zhats = []
Zbounds = []

for i in range(len(current)):
    zhat, zbound = LIB1_SPKF.iter(current[i], voltage[i])
    # cv2.imshow('My Battery', np.hstack([plot_SOC(int(soc[i]*100)), plot_SOC(int(zhat*100))]))
    # cv2.waitKey(1)

    # store data
    Ztrues.append(soc[i].item())
    Zhats.append(zhat.item())
    Zbounds.append(zbound.item())

Ztrues = np.array(Ztrues)
Zhats = np.array(Zhats)
Zbounds = np.array(Zbounds)
# maxIter = len(Zhats)

# compute errors
print(f'RMS SOC estimation error = {np.sqrt(np.mean((100*(Ztrues[:-1]-Zhats))**2))}')
print(f'Percent of time error outside bounds = {(np.sum(abs(Ztrues[:-1]-Zhats)>Zbounds)/len(Ztrues[:-1]))*100}')

# plot diagrams
plt.figure()
plt.subplot(2,2,1)
plt.plot(time,current)
plt.grid()
plt.title('Current')
# plt.xlabel('Time (min)')
plt.ylabel('Current (A)')

plt.subplot(2,2,2)
plt.plot(time/60,voltage)
plt.grid()
plt.title('Voltage')
# plt.xlabel('Time (min)')
plt.ylabel('Votltage (V)')

plt.subplot(2,2,3)
plt.plot(time/60,100*Ztrues[:-1],color=(0,0.8,0))
plt.plot(time/60,100*Zhats,color=(0,0,1),linestyle='dashed')
plt.fill_between(time/60, 100*(Zhats+Zbounds), 100*(Zhats-Zbounds), alpha=0.3)
plt.grid()
plt.legend(['Truth','Estimate','Bounds'])
plt.title('SOC estimation using SPKF')
plt.xlabel('Time (min)')
plt.ylabel('SOC (%)')

plt.subplot(2,2,4)
estErr = Ztrues[:-1]-Zhats 
plt.plot(time/60, 100*estErr)
plt.fill_between(time/60, 100*Zbounds, -100*Zbounds, alpha=0.3)
plt.grid()
plt.legend(['Estimation error','Bounds'])
plt.title('SOC estimation errors using SPKF')
plt.xlabel('Time (min)') 
plt.ylabel('SOC error (%)')
plt.show()
    

