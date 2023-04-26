import numpy as np 
from scipy.linalg import block_diag, cholesky
from copy import copy

class BarDelta_SPKF:
    def __init__(self, BatteryPack, SigmaX0, SigmaW, SigmaV):
        # battery pack information
        self.SCMpack = copy(BatteryPack)
        self.NUM_CELL = len(self.SCMpack)
        self.dt = self.SCMpack[0].dt
        self.set_bar_parameters()

        # store model
        ir0   = 0                        
        hk0   = 0                        
        SOC0  = 0.75

        # initial state
        self.xhat  = np.vstack([ir0, hk0, SOC0, 0, 0, 0]) 

        # variable indexes
        self.iRInd = 0
        self.hInd =  1
        self.zInd =  2
        self.ibInd = 3
        self.R0Ind = 4
        self.QinvInd = 5

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
        self.Wm = Wmx.reshape(-1,1)                     # mean
        self.Wc = Wcx.reshape(-1,1)                     # covar
        
        # CDKF delta weights
        self.dNx = 1                                    # one state per delta filter, for estimating SOC
        self.dNw = 1                                    # one process-noise per delta filter
        self.dNa = self.dNx+self.dNw                    # augmented length for delta filters

        dWmx = np.zeros(2*self.dNa+1)
        dWmx[0] = (self.h**2-self.dNa)/(self.h**2)      # weighting factors when computing mean
        dWmx[1:] = 1/(2*self.h**2)                      # and covariance
        dWcx=dWmx                       
        self.dWm = dWmx.reshape(-1,1)                   # mean
        self.dWc = dWcx.reshape(-1,1)                   # covar
        
        # previous value of current
        self.priorI = 0
        self.Qbump = 5

        # reserving storage
        self.dz = np.zeros(self.NUM_CELL)
        self.Sdz = SigmaX0[self.zInd, self.zInd]*np.ones(self.NUM_CELL)
        
        self.dR0 = self.R0Param_bar - np.arange(1.3, 1, -0.1)*1e-3
        self.dQinv = 1/np.arange(25, 29, 1) - self.QinvParam_bar

    def iter_bar(self, ik, vk):
        # Step 1a : State estimate time update
        #       - Create xhatminus augmented SigmaX points
        #       - Extract xhatminus state SigmaX points
        #       - Compute weighted average xhatminus(k)

        # Step 1a-1: Create augmented SigmaX and xhat
        sigmaXa = block_diag(self.SigmaX, self.SigmaW, self.SigmaV)
        sigmaXa = np.real(cholesky(sigmaXa, lower=True))
        xhata = np.vstack([self.xhat, np.zeros((self.Nw+self.Nv,1))])
        
        # Step 1a-2: Calculate SigmaX points (strange indexing of xhat a to
        Xa = xhata.reshape(self.Na,1) + self.h*np.hstack([np.zeros((self.Na, 1)), sigmaXa, -sigmaXa])
        
        # Step 1a-3: Time update from last iteration until now
        Xx = self.stateEqn(self.priorI, Xa[self.Nx:self.Nx+self.Nw,:], Xa[:self.Nx,:])
        Xx = np.vstack(Xx)
        xhat = Xx@self.Wm
        
        # Step 1b: Error covariance time update
        #           - Compute weighted covariance sigmaminus(k)
        #           (strange indexing of xhat to avoid "repmat" call)
        SigmaX = (Xx - xhat)@np.diag(self.Wc.ravel())@(Xx - xhat).T
        
        # Step 1c: Output estimate
        #           - Compute weighted output estimate yhat(k)
        Y = self.outputEqn(ik, Xa[self.Nx+self.Nw:,:], Xx)
        yhat = Y@self.Wm
        
        # Step 2a: Estimator gain matrix
        SigmaXY = (Xx - xhat)@np.diag(self.Wc.ravel())@(Y - yhat).T
        SigmaY  = (Y - yhat)@np.diag(self.Wc.ravel())@(Y - yhat).T
        L = SigmaXY/SigmaY
        
        # Step 2b: State estimate measurement update
        r = np.mean(vk) - yhat  # residual.  Use to check for sensor errors...
        if r**2 > 100*SigmaY: L=0
        
        xhat = xhat + L*r 
        xhat[self.hInd] = np.clip(xhat[self.hInd], -1, 1)
        xhat[self.zInd] = np.clip(xhat[self.zInd], -0.05, 1.05)
        
        # Step 2c: Error covariance measurement update
        SigmaX = SigmaX - L*SigmaY*L.T
        _,S,V = np.linalg.svd(SigmaX)
        HH = V.T@np.diag(S)@V
        SigmaX = (SigmaX + SigmaX.T + HH + HH.T)/4 # Help maintain robustness
        SigmaX = np.abs(SigmaX)
        # Q-bump code
        if r**2>4*SigmaY: # bad voltage estimate by 2-SigmaX, bump Q 
            print('Bumping sigmax\n')
            SigmaX[self.zInd, self.zInd] = SigmaX[self.zInd, self.zInd]*self.Qbump

        # self.priorI = ik
        # Save data in spkfData structure for next time...
        self.SigmaX = SigmaX
        self.xhat = xhat

        self.ib = xhat[self.ibInd]
        ibbnd = 3*np.sqrt(SigmaX[self.ibInd, self.ibInd])
        
        zk = xhat[self.zInd]
        zkbnd = 3*np.sqrt(SigmaX[self.zInd, self.zInd])
        return zk,zkbnd,xhat[self.ibInd],xhat[self.R0Ind],xhat[self.QinvInd]

        

    def iter_delta(self, ik, vk):    
        # Updating the delta-SOC SPKFs 
        # delta-SOC
        for thecell in range(self.NUM_CELL):
            # Step 1a - State prediction time update
            sigmaXa  = block_diag(np.sqrt(np.maximum(0,self.Sdz[thecell])), np.sqrt(self.SigmaW[0,0]))
            
            # sigmaXa  = np.real(cholesky(sigmaXa , lower=True))
            xhata  = np.vstack([self.dz[thecell], np.zeros((self.dNw,1))])
            
            Xa = xhata.reshape(self.dNa,1) + self.h*np.hstack([np.zeros((self.dNa, 1)), sigmaXa, -sigmaXa])
            # priorI is updated sooner!!!
            Xx = self.deltazStateEqn(thecell, self.priorI - self.ib, Xa[self.dNx:self.dNx+self.dNw,:], Xa[:self.dNx,:])
            
            Xx = np.vstack(Xx)
            xhat  = Xx@self.dWm
            
            # Step 1b - Do error covariance time update
            SigmaX = (Xx - xhat)@np.diag(self.dWc.ravel())@(Xx - xhat).T
            
            # Step 1c - output estimate
            Y = self.deltazOutputEqn(thecell, ik - self.ib, Xa[self.dNx:self.dNx+self.dNw,:], Xx)
            yhat = Y@self.dWm

            # Step 2a - Estimator gain matrix
            SigmaXY = (Xx - xhat)@np.diag(self.dWc.ravel())@(Xx - xhat).T
            SigmaY = (Y - yhat)@np.diag(self.dWc.ravel())@(Y - yhat).T + self.SigmaV
            L = SigmaXY/SigmaY
            
            # Step 2b - State estimate measurement update
            self.dz[thecell] = xhat + L*(vk[thecell] - yhat) 

            # Step 2c - Error covariance measurement update
            self.Sdz[thecell] = SigmaX - L*SigmaY*L.T
            
        # print(self.Sdz[3])
        self.priorI = ik


    def stateEqn(self, current, xnoise=0, oldState=None):

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
        # current = np.where(current<0, current*self.etaParam, current)
        current = current + xnoise[0,:]
        
        Ah = np.exp(-abs(current*self.GParam*self.dt*Qinv_k/3600))  # hysteresis factor
        Arc = np.diag(np.exp(-self.dt/abs(self.RCParam)))
        Brc = 1-(np.exp(-self.dt/abs(self.RCParam)))

        iR_k1 = Arc@iR_k + Brc*current
        h_k1 = Ah*h_k - (1-Ah)*np.sign(current)
        z_k1 = z_k - (self.dt*Qinv_k/3600)*current
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
        
        sik = 0
        voltage = self.OCVfromSOC(z_k) + self.MParam*h_k + self.M0Param*sik - self.RParam*iR_k - R0_k*(current-ib_k) + ynoise
        return voltage

    def deltazStateEqn(self, thecell, pre_current, xnoise, old_deltaz):
        pre_current = pre_current + xnoise
        new_deltaz = old_deltaz - (pre_current)*self.dt*self.dQinv[thecell]/3600 # important #######################33
        return new_deltaz

    def deltazOutputEqn(self, thecell, current, ynoise, dz):
        current = current + ynoise
        voltage = self.OCVfromSOC(self.xhat[self.zInd]+dz) \
                + self.MParam*self.xhat[self.hInd] \
                - self.RParam*self.xhat[self.iRInd] \
                - (self.R0Param_bar+self.dR0[thecell])*current

        return voltage

### ok functions

    def set_bar_parameters(self):
        self.etaParam   = self.SCMpack[0].etaParam
        self.GParam     = self.SCMpack[0].GParam
        self.M0Param    = self.SCMpack[0].M0Param
        self.MParam     = self.SCMpack[0].MParam
        self.RCParam    = self.SCMpack[0].RCParam
        self.RParam     = self.SCMpack[0].RParam
        self.OCVfromSOC = self.SCMpack[0].OCVfromSOC
        self.SOCfromOCV = self.SCMpack[0].SOCfromOCV
        self.QinvParam_bar = np.mean([1/(self.SCMpack[i].QParam) for i in range(len(self.SCMpack))])
        self.R0Param_bar   = np.mean([self.SCMpack[i].R0Param for i in range(len(self.SCMpack))])