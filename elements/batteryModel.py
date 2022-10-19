import numpy as np
from scipy import interpolate
from mat4py import loadmat

class LithiumIonBattery():
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
        self.z_k = 1
        self.iR_k = np.zeros((1,1))
        self.h_k = 0
        
    def update_state(self, current):
        eta=self.etaParam if current<0 else 1
        Ah = np.exp(-abs(eta*current*self.GParam*self.dt/(3600*self.QParam)))  # hysteresis factor
        Arc = np.diag(np.exp(-self.dt/abs(self.RCParam)))
        Brc = 1-(np.exp(-self.dt/abs(self.RCParam)))

        z_k1 = self.z_k - (eta*self.dt/(3600*self.QParam))*current
        iR_k1 = Arc@self.iR_k + Brc*current
        h_k1 = Ah*self.h_k - (1-Ah)*np.sign(current)

        z_k1 = np.clip(z_k1, -0.05, 1.05)
        h_k1 = np.clip(h_k1, -1, 1)

        if abs(current)>self.QParam/100: self.sik = np.sign(current)
        voltage = self.OCVfromSOC(self.z_k) + self.MParam*self.h_k + self.M0Param*self.sik - self.RParam*self.iR_k - self.R0Param*current

        self.z_k = z_k1
        self.iR_k = iR_k1
        self.h_k = h_k1

        return voltage.item()


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
