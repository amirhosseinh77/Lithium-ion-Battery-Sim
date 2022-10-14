from elements.batteryModel import LithiumIonBattery
from elements.plots import plot_SOC
import cv2


LIB1 = LithiumIonBattery('models/PANmodel.mat', T=25, dt=1)

Z = []
V = []
while True:
    if LIB1.z_k > 0.8: i = 100
    if LIB1.z_k < 0.2: i = -100
    
    v = LIB1.update_state(i)
    cv2.imshow('My Battery', plot_SOC(int(LIB1.z_k*100)))
    cv2.waitKey(1)
    # Z.append(LIB1.z_k)
    # V.append(v)

