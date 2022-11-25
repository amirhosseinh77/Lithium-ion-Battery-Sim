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
        self.Wm = Wmx.reshape(-1,1) # mean
        self.Wc = Wcx.reshape(-1,1) # covar

        # CDKF delta weights
        self.dNx = 1                             # one state per delta filter, for estimating SOC
        self.dNw = 1                             # one process-noise per delta filter
        self.dNv = 1
        self.dNa = self.dNx+self.dNw+self.dNv    # augmented length for delta filters

        dWmx = np.zeros(2*dNa+1)
        dWmx[0] = (h**2-dNa)/(h**2)              # weighting factors when computing mean
        dWmx[1:] = 1/(2*h**2)                    # and covariance
        dWcx=dWmx                       
        self.dWm = dWmx.reshape(-1,1)               # mean
        self.dWc = dWcx.reshape(-1,1)               # covar

        # previous value of current
        self.priorI = 0
        self.Qbump = 5

        # reserving storage
        self.dz = np.zeros(self.NUM_CELL)
        self.Sdz = SigmaX0[self.zInd, self.zInd]*np.ones(self.NUM_CELL)

        self.dR0 = np.zeros(self.NUM_CELL)
        self.SdR0 = SigmaX0[self.R0Ind, self.R0Ind]*np.ones(self.NUM_CELL)

        self.dQinv = np.zeros(self.NUM_CELL)
        self.SdQinv = SigmaX0[self.QinvInd, self.QinvInd]*np.ones(self.NUM_CELL)

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

        # Q-bump code
        if r**2>4*SigmaY: # bad voltage estimate by 2-SigmaX, bump Q 
            print('Bumping sigmax\n')
            SigmaX[self.zInd, self.zInd] = SigmaX[self.zInd, self.zInd]*self.Qbump

        # Save data in spkfData structure for next time...
        self.priorI = ik
        self.SigmaX = SigmaX
        self.xhat = xhat

        ib = xhat[self.ibInd]
        ibbnd = 3*np.sqrt(SigmaX[self.ibInd, self.ibInd])

        zk = xhat[self.zInd]
        zkbnd = 3*np.sqrt(SigmaX[self.zInd, self.zInd])
        # return zk,zkbnd

    def iter_delta(self, ik, vk):
        # Updating the delta-SOC, delta-R0 & delta-Qinv SPKFs
        # delta-SOC
        for thecell in range(self.NUM_CELL):
            # Step 1a - State prediction time update
            sigmaXa  = block_diag(self.Sdz[thecell], SigmaW[1,1], SigmaV)
            sigmaXa  = np.real(cholesky(sigmaXa , lower=True))
            xhata  = np.vstack([self.dz[thecell], np.zeros((dNw+Nv,1))])

            Xa = xhata.reshape(self.dNa,1) + h*np.hstack([np.zeros((self.dNa, 1)), sigmaXa, -sigmaXa])
            # priorI is updated sooner!!!
            Xx = self.deltazStateEqn(priorI, xnoise=Xa[self.dNx:self.dNx+self.dNw,:], deltaz_k=Xa[:self.dNx,:], celldQinv=0)  ###??? cellXa[Nx:Nx+Nw,:], Xa[:Nx,:]
            Xx = np.vstack(Xx)
            xhat  = Xx@self.dWm

            # Step 1b - Do error covariance time update
            SigmaX = (Xx - xhat)@np.diag(dWc.ravel())@(Xx - xhat).T

            # Step 1c - output estimate
            Y = self.deltazOutputEqn(priorI, zk, Xx, ynoise=Xa[dNx+dNw:,:])
            yhat = Y@self.dWm

            # Step 2a - Estimator gain matrix
            SigmaXY = (Xx - xhat)@np.diag(dWc.ravel())@(Xx - xhat).T
            SigmaY = (Y - yhat)@np.diag(dWc.ravel())@(Y - yhat).T
            L = SigmaXY/SigmaY

            # Step 2b - State estimate measurement update
            self.dz[thecell] = xhat + L*(vk[thecell] - yhat) 

            # Step 2c - Error covariance measurement update
            self.Sdz[thecell] = SigmaX - L*SigmaY*L.T

        # delta-R0
        for thecell in range(self.NUM_CELL):
            R0 = self.dR0[thecell]
            SdR0 = self.SdR0[thecell]

            # Step 1a - State prediction time update
            cellSxa = block_diag(cellSdR0[thecell], SigmaW[2,2], SigmaV)
            cellSxa = np.real(cholesky(cellSxa, lower=True))
            cellXa = np.vstack([celldR0[thecell], np.zeros((dNw+Nv,1))])
            cellXa = cellXa.reshape(dNa,1) + h*np.hstack([np.zeros((dNa, 1)), cellSxa, -cellSxa])
            cellXx = deltaR0StateEqn(xnoise=cellXa[dNx:dNx+dNw,:], deltaR0_k=cellXa[:dNx,:])  ###??? cellXa[Nx:Nx+Nw,:], Xa[:Nx,:]
            cellXx = np.vstack(cellXx)
            celldR0[thecell] = cellXx@dWm

            # Step 1b - Do error covariance time update
            cellSdR0[thecell] = (cellXx - celldR0[thecell])@np.diag(dWc.ravel())@(cellXx - celldR0[thecell]).T

            # Step 1c - output estimate
            cellY = deltaR0OutputEqn(priorI, zk, cellSdz[thecell], ynoise=cellXa[dNx+dNw:,:])
            cellyhat = cellY@dWm

            # Step 2a - Estimator gain matrix
            cellSxy = (cellXx - celldR0[thecell])@np.diag(dWc.ravel())@(cellXx - celldR0[thecell]).T
            cellSy  = (cellY - cellyhat)@np.diag(dWc.ravel())@(cellY - cellyhat).T
            cellL = cellSxy/cellSy

            # Step 2b - State estimate measurement update
            celldR0[thecell] = celldR0[thecell] + cellL*(vkk[thecell] - cellyhat) 

            # Step 2c - Error covariance measurement update
            cellSdR0[thecell] = cellSdR0[thecell] - cellL*cellSy*cellL.T

        # delta-Qinv
        for thecell in range(NUM_CELL):
            dQinv = self.dQinv[thecell]
            SdQinv = self.SdQinv[thecell]

            # Step 1a - State prediction time update
            cellSxa = block_diag(cellSdQinv[thecell], SigmaW[3,3], SigmaV)
            cellSxa = np.real(cholesky(cellSxa, lower=True))
            cellXa = np.vstack([celldQinv[thecell], np.zeros((dNw+Nv,1))])
            cellXa = cellXa.reshape(dNa,1) + h*np.hstack([np.zeros((dNa, 1)), cellSxa, -cellSxa])
            cellXx = deltaQinvStateEqn(xnoise=cellXa[dNx:dNx+dNw,:], deltaQinv_k=cellXa[:dNx,:])  ###??? cellXa[Nx:Nx+Nw,:], Xa[:Nx,:]
            cellXx = np.vstack(cellXx)
            celldQinv[thecell] = cellXx@dWm

            # Step 1b - Do error covariance time update
            cellSdQinv[thecell] = (cellXx - celldQinv[thecell])@np.diag(dWc.ravel())@(cellXx - celldQinv[thecell]).T

            # Step 1c - output estimate
            cellY = deltaQinvOutputEqn(priorI, zk, cellXx, ynoise=cellXa[dNx+dNw:,:])
            cellyhat = cellY@dWm

            # Step 2a - Estimator gain matrix
            cellSxy = (cellXx - celldQinv[thecell])@np.diag(dWc.ravel())@(cellXx - celldQinv[thecell]).T
            cellSy  = (cellY - cellyhat)@np.diag(dWc.ravel())@(cellY - cellyhat).T
            cellL = cellSxy/cellSy

            # Step 2b - State estimate measurement update
            celldQinv[thecell] = celldQinv[thecell] + cellL*(vkk[thecell] - cellyhat) 

            # Step 2c - Error covariance measurement update
            cellSdQinv[thecell] = cellSdQinv[thecell] - cellL*cellSy*cellL.T


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
