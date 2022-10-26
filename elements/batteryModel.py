import numpy as np
from scipy import interpolate
from mat4py import loadmat

class LithiumIonBattery:
    def __init__(self, model_path, T, dt):
        data = loadmat(model_path)
        self.name = data['model']['name']
        self.temps = data['model']['temps']
        self.etaParam = data['model']['etaParam'][self.temps.index(T)]
        self.QParam = data['model']['QParam'][self.temps.index(T)]
        self.GParam = data['model']['GParam'][self.temps.index(T)]
        self.M0Param = data['model']['M0Param'][self.temps.index(T)]
        self.MParam = data['model']['MParam'][self.temps.index(T)]
        self.R0Param = data['model']['R0Param'][self.temps.index(T)]
        self.RCParam = np.array(data['model']['RCParam'][self.temps.index(T)])
        self.RParam = np.array(data['model']['RParam'][self.temps.index(T)])
        self.OCVfromSOC = make_OCVfromSOCtemp(data, T)

        self.sik = 0
        self.dt = dt
        self.z_k = np.ones((1,1))
        self.iR_k = np.zeros((1,1))
        self.h_k = np.zeros((1,1))

    def stateEqn(self, current, xnoise=0, oldState=None):
        if oldState is not None: 
            (z_k, iR_k, h_k) = oldState
            z_k =  z_k.reshape(1,-1)
            iR_k = iR_k.reshape(1,-1)
            h_k =  h_k.reshape(1,-1)
        else:
            z_k =  self.z_k.reshape(1,-1)
            iR_k = self.iR_k.reshape(1,-1)
            h_k =  self.h_k.reshape(1,-1)

        prisik = self.sik
        current = current + xnoise
        eta = np.where(current<0, self.etaParam, 1) #self.etaParam if current<0 else 1
        Ah = np.exp(-abs(eta*current*self.GParam*self.dt/(3600*self.QParam)))  # hysteresis factor
        Arc = np.diag(np.exp(-self.dt/abs(self.RCParam)))
        Brc = 1-(np.exp(-self.dt/abs(self.RCParam)))

        z_k1 = z_k - (eta*self.dt/(3600*self.QParam))*current
        iR_k1 = Arc@iR_k + Brc*current
        h_k1 = Ah*h_k - (1-Ah)*np.sign(current)
        z_k1 = np.clip(z_k1, -0.05, 1.05)
        h_k1 = np.clip(h_k1, -1, 1)

        # if abs(current)>self.QParam/100: self.sik = np.sign(current)
        self.sik = np.where(abs(current)>self.QParam/100, np.sign(current), prisik)
        newState = (z_k1, iR_k1, h_k1)
        return newState

    def outputEqn(self, current, ynoise=0, state=None):
        if state is not None: 
            (z_k, iR_k, h_k) = state
            z_k =  z_k.reshape(1,-1)
            iR_k = iR_k.reshape(1,-1)
            h_k =  h_k.reshape(1,-1)
        else:
            z_k =  self.z_k
            iR_k = self.iR_k
            h_k =  self.h_k

        current = current + ynoise
        voltage = self.OCVfromSOC(z_k) + self.MParam*h_k + self.M0Param*self.sik - self.RParam*iR_k - self.R0Param*current + ynoise
        return voltage

    def updateState(self, newState):
        (z_k1, iR_k1, h_k1) = newState
        self.z_k = z_k1
        self.iR_k = iR_k1
        self.h_k = h_k1

def make_OCVfromSOCtemp(data, T):
    SOC = np.array(data['model']['SOC'])
    OCV0 = np.array(data['model']['OCV0'])
    OCVrel = np.array(data['model']['OCVrel'])
    OCV = OCV0 + T*OCVrel
    OCVfromSOC = interpolate.interp1d(SOC, OCV, fill_value="extrapolate")
    return OCVfromSOC

def make_dOCVfromSOCtemp(data, T):
    SOC = np.array(data['model']['SOC'])
    OCV0 = np.array(data['model']['OCV0'])
    OCVrel = np.array(data['model']['OCVrel'])
    OCV = OCV0 + T*OCVrel

    dZ = SOC[1] - SOC[0]
    dUdZ = np.diff(OCV)/dZ
    dOCV = (np.append(dUdZ[0],dUdZ) + np.append(dUdZ,dUdZ[-1]))/2
    dOCVfromSOC = interpolate.interp1d(SOC, dOCV, fill_value="extrapolate")
    return dOCVfromSOC
