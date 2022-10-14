import cv2
import numpy as np

def plot_SOC(SOC):
    img = np.ones((300,300,3), dtype='uint8')*255
    img = cv2.rectangle(img, (95,45), (205,265), 0, 10) # H: 50-260 / W: 100-200
    img = cv2.rectangle(img, (130,30), (170,45), 0, -1)
    soc = np.array(range(0,SOC+1))
    soc = np.append(0,soc)
    green = (soc*255/100).astype(int)
    red = 255-green
    y = (-210/100*soc + 260).astype(int)
    for i in range(1,len(soc)): img[y[i]:y[i-1],100:200] = [0,green[i],red[i]] 
    return img