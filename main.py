from elements.batteryModel import LithiumIonBattery, make_OCVfromSOCtemp, make_dOCVfromSOCtemp
from elements.plots import plot_SOC
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat

############################## OCV and dOCV ##############################
data = loadmat('models/PANmodel.mat')
OCVfromSOC = make_OCVfromSOCtemp(data, 25)
dOCVfromSOC = make_dOCVfromSOCtemp(data, 25)
xnew = np.arange(0,1,0.01)
fnew = OCVfromSOC(xnew)
dfnew = dOCVfromSOC(xnew)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(data['model']['SOC'],data['model']['OCV0'])
plt.plot(xnew,fnew,linestyle='--')
plt.grid()
plt.subplot(1,2,2)
plt.plot(xnew,dfnew)
plt.grid()
plt.show()


############################## Simulate Model ##############################
LIB1 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=1)

Z = []
V = []
while True:
    if LIB1.z_k > 0.8: i = 1000
    if LIB1.z_k < 0.2: i = -1000
    
    newState = LIB1.stateEqn(i)
    voltage = LIB1.outputEqn(i)
    LIB1.updateState(newState)
    cv2.imshow('My Battery', plot_SOC(int(LIB1.z_k*100)))
    cv2.waitKey(1)
    # Z.append(LIB1.z_k)
    # V.append(v)

