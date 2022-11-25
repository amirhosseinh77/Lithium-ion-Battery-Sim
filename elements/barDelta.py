import numpy as np 
from scipy.linalg import block_diag, cholesky
from copy import copy

class BarDelta_SPKF:
    def __init__(self, BatteryPack, SigmaX0, SigmaV, SigmaW):
        # battery pack information
        self.SCMpack = copy(BatteryPack)
        self.NUM_CELL = len(self.pack)

        # store model
        ir0   = 0                        
        hk0   = 0                        
        SOC0  = np.mean([self.SCMpack[i].SOCfromOCV(self.SCMpack[i].outputEqn(0)) for i in range(self.NUM_CELL)])

        # initial state
        self.xhat  = np.vstack([ir0,hk0,SOC0,0,0,0]) 

        # # variable indexes
        # self.iR
        # iR_k = iR_k.reshape(1,-1)
        # h_k =  h_k.reshape(1,-1)
        # z_k =  z_k.reshape(1,-1)
        # ib_k =  ib_k.reshape(1,-1)
        # R0_k =  R0_k.reshape(1,-1)
        # Qinv_k =  Qinv_k.reshape(1,-1)

        # Covariance values
        self.SigmaX = SigmaX0
        self.SigmaW = SigmaW
        self.SigmaV = SigmaV

        # SPKF specific parameters
        self.Nx = self.SigmaX.shape[0]
        self.Nw = self.SigmaW.shape[0]
        self.Nv = self.SigmaV.shape[0]
        self.Na = self.Nx+self.Nw+self.Nv

        # CDKF bar weights
        Wmx = np.zeros(2*self.Na+1)
        self.h = np.sqrt(3)
        Wmx[0] = (self.h**2-self.Na)/(self.h**2)
        Wmx[1:] = 1/(2*self.h**2) 
        Wcx=Wmx
        self.Wm = Wmx.reshape(-1,1) # mean
        self.Wc = Wcx.reshape(-1,1) # covar

        # CDKF delta weights
        self.dNx = 1                             # one state per delta filter, for estimating SOC
        self.dNw = 1                             # one process-noise per delta filter
        self.dNv = 1
        self.dNa = self.dNx+self.dNw+self.dNv    # augmented length for delta filters

        dWmx = np.zeros(2*dNa+1)
        dWmx[0] = (h**2-dNa)/(h**2)                 # weighting factors when computing mean
        dWmx[1:] = 1/(2*h**2)                       # and covariance
        dWcx=dWmx                       
        self.dWm = dWmx.reshape(-1,1)               # mean
        self.dWc = dWcx.reshape(-1,1)               # covar

        # previous value of current
        self.priorI = 0
        self.Qbump = 5

    def iter(self, ik, vk):
        pass


    def stateEqn(self, current, xnoise=0, oldState=None):
        dt = 1
        if oldState is not None: 
            (iR_k, h_k, z_k, ib_k, R0_k, Qinv_k) = oldState
            iR_k = iR_k.reshape(1,-1)
            h_k =  h_k.reshape(1,-1)
            z_k =  z_k.reshape(1,-1)
            ib_k =  ib_k.reshape(1,-1)
            R0_k =  R0_k.reshape(1,-1)
            Qinv_k =  Qinv_k.reshape(1,-1)
            
        # if abs(current)>1/(100*Qinv_k): sik = np.sign(current)
        sik = 0
        current = current - ib_k
        current = np.where(current<0, current*etaParam, current)
        current = current + xnoise[0,:]
        
        Ah = np.exp(-abs(current*GParam*dt*Qinv_k/3600))  # hysteresis factor
        Arc = np.diag(np.exp(-dt/abs(RCParam)))
        Brc = 1-(np.exp(-dt/abs(RCParam)))

        iR_k1 = Arc@iR_k + Brc*current
        h_k1 = Ah*h_k - (1-Ah)*np.sign(current)
        z_k1 = z_k - (dt*Qinv_k/3600)*current
        ib_k1 = ib_k + xnoise[1,:]
        R0_k1 = R0_k + xnoise[2,:]
        Qinv_k1 = Qinv_k + xnoise[3,:]

        h_k1 = np.clip(h_k1, -1, 1)
        z_k1 = np.clip(z_k1, -0.05, 1.05)

        newState = (iR_k1, h_k1, z_k1, ib_k1, R0_k1, Qinv_k1)
        return newState


    # Calculate cell output voltage for all of state vectors in xhat
    def outputEqn(self, current, ynoise=0, state=None):
        if state is not None: 
            (iR_k, h_k, z_k, ib_k, R0_k, Qinv_k) = state
            iR_k = iR_k.reshape(1,-1)
            h_k =  h_k.reshape(1,-1)
            z_k =  z_k.reshape(1,-1)
            ib_k =  ib_k.reshape(1,-1)
            R0_k =  R0_k.reshape(1,-1)
            Qinv_k =  Qinv_k.reshape(1,-1)

        voltage = LIB1.OCVfromSOC(z_k) + MParam*h_k + M0Param*LIB1.sik - RParam*iR_k - R0Param*(current-ib_k) + ynoise
        return voltage


    def deltazStateEqn(self, current, xnoise=0, deltaz_k=0, celldQinv=0):
        dt = 1
        current = current + xnoise
        deltaz_k1 = deltaz_k - (current)*dt*celldQinv/3600
        return deltaz_k1

    def deltazOutputEqn(self, current, z_k, deltaz_k1, ynoise=0):
        h_k = -1.16634495e-02
        iR_k = 3.37375367e-01
        deltaR0_k1 = 0
        ib_k = 0
        voltage = LIB1.OCVfromSOC(z_k+deltaz_k1) + MParam*h_k + M0Param*LIB1.sik - RParam*iR_k - (R0Param+deltaR0_k1)*(current-ib_k) + ynoise
        return voltage

    def deltaR0StateEqn(self, xnoise=0,deltaR0_k=0):
        n_deltaR0 = xnoise
        deltaR0_k1 = deltaR0_k + n_deltaR0
        return deltaR0_k1

    def deltaR0OutputEqn(self, current, z_k, deltaz_k1=0, ynoise=0):
        deltaR0_k1 = 0
        ib_k = 0

        voltage = LIB1.OCVfromSOC(z_k+deltaz_k1) - (R0Param+deltaR0_k1)*(current-ib_k) + ynoise
        return voltage

    def deltaQinvStateEqn(self, xnoise=0,deltaQinv_k=0):
        n_deltaQinv = xnoise
        deltaQinv_k1 = deltaQinv_k + n_deltaQinv
        return deltaQinv_k1

    def deltaQinvOutputEqn(self, current_k, deltaz_k1=0, deltaz_k=0, ynoise=0):
        deltaR0_k1 = 0
        ib_k = 0
        Qinv_k =1
        deltaQinv_k = 0
        dt=1
        dk = (deltaz_k1 - deltaz_k) - (current_k-ib_k)*dt*(Qinv_k+deltaQinv_k)/3600 + ynoise
        return dk
