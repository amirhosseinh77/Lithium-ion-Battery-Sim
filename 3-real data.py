from elements.batteryModel import LithiumIonBattery, make_OCVfromSOCtemp, make_dOCVfromSOCtemp
from elements.estimator import SPKF
from elements.plots import plot_SOC
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from scipy.linalg import block_diag

data = loadmat('./PANdata_P25.mat')
time    = np.array(data['DYNData']['script1']['time'])  
deltat = time[1]-time[0]
time    = time-time[0] # start time at 0
current = np.array(data['DYNData']['script1']['current']) # discharge > 0; charge < 0.
voltage = np.array(data['DYNData']['script1']['voltage'])
soc     = np.array(data['DYNData']['script1']['soc'])

plt.figure(figsize=(12,4))
plt.subplot(2,1,1)
plt.plot(time,current)
plt.grid()
plt.subplot(2,1,2)
plt.plot(time,voltage)
plt.grid()
plt.show()
############################## Simulate Model ##############################
LIB1 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=deltat)
        
SigmaX = block_diag(1e2,1e-2,1e-3)  # uncertainty of initial state
SigmaV = 3e-1  # Uncertainty of voltage sensor, output equation
SigmaW = 4e0   # Uncertainty of current sensor, state equation
LIB1_SPKF = SPKF(LIB1, SigmaX, SigmaV, SigmaW)

Ztrues = [soc[0]]
Zhats = []
Zbounds = []
V = []

for i in range(len(current)):
    zhat, zbound = LIB1_SPKF.iter(current[i], voltage[i])

    # cv2.imshow('My Battery', np.hstack([plot_SOC(int(soc[i]*100)), plot_SOC(int(zhat*100))]))
    # cv2.waitKey(1)

    # store data
    Ztrues.append(soc[i].item())
    Zhats.append(zhat.item())
    Zbounds.append(zbound.item())
    # V.append(v)


Ztrues = np.array(Ztrues)
Zhats = np.array(Zhats)
Zbounds = np.array(Zbounds)
maxIter = len(Zhats)

# plot diagrams
plt.figure(figsize=(18,6))
plt.subplot(2,1,1)
plt.plot(np.arange(maxIter),Ztrues[:maxIter],color=(0,0.8,0))
plt.plot(np.arange(maxIter),Zhats,color=(0,0,1),linestyle='dashed')
plt.fill_between(np.arange(maxIter), Zhats+Zbounds, Zhats-Zbounds, alpha=0.3)
plt.grid()
plt.legend(['true','estimate','bounds'])
plt.title('Sigma-point Kalman filter in action')
plt.xlabel('Iteration')
plt.ylabel('SOC')

plt.subplot(2,1,2)
estErr = Ztrues[:maxIter]-Zhats 
plt.plot(np.arange(maxIter), estErr)
plt.fill_between(np.arange(maxIter), Zbounds, -Zbounds, alpha=0.3)
plt.grid()
plt.legend(['Error','bounds'])
plt.title('SPKF Error with bounds')
plt.xlabel('Iteration') 
plt.ylabel('Estimation Error')
plt.show()
    

