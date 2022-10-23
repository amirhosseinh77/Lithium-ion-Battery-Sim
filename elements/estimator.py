import numpy as np 
from scipy.linalg import block_diag, cholesky
from copy import copy

class SPKF:
    def __init__(self, target_model, SigmaX0, SigmaV, SigmaW):
        # store model
        self.model = copy(target_model)
        ir0   = 0                        
        hk0   = 0                        
        SOC0  = self.model.OCVfromSOC(ir0)
        self.xhat  = np.vstack([ir0,hk0,SOC0]) # initial state

        # Covariance values
        self.SigmaX = SigmaX0
        self.SigmaV = SigmaV
        self.SigmaW = SigmaW

        # SPKF specific parameters
        self.Nx = SigmaX0.shape[0]
        self.Nw = 1
        self.Nv = 1
        self.Na = Nx+Nw+Nv

        Wmx = np.zeros(2*self.Na+1)
        h = np.sqrt(3)
        Wmx[0] = (h**2-self.Na)/(h**2)
        Wmx[1:] = 1/(2*h**2) 
        Wcx=Wmx
        self.Wm = Wmx # mean
        self.Wc = Wcx # covar
        self.Qbump = 5

        # previous value of current
        self.priorI = 0
        
    def iter(self, ik, vk):
        # Step 1a-1: Create augmented xhat and SigmaX
        sigmaXa = block_diag(self.SigmaX, self.SigmaW, self.SigmaV)
        sigmaXa = np.real(cholesky(sigmaXa, lower=True))
        xhata = np.vstack([self.xhat, np.zeros((self.Nw+self.Nv,1))])

        # Step 1a-2: Calculate SigmaX points
        Xa = xhata.reshape(self.Na,1) + h*np.hstack([np.zeros((self.Na, 1)), sigmaXa, -sigmaXa])

        # Step 1a-3: Time update from last iteration until now
        Xx = LIB1.stateEqn(self.priorI, Xa[Nx:Nx+Nw,:], Xa[:Nx,:])
        Xx = np.vstack(Xx)
        xhat = np.sum(self.Wm*Xx, axis=-1, keepdims=True)

        # Step 1b: Error covariance time update
        #          - Compute weighted covariance sigmaminus(k)
        #            (strange indexing of xhat to avoid "repmat" call)
        SigmaX = np.sum(self.Wc*(Xx - xhat)**2, axis=-1)

        # Step 1c: Output estimate
        #          - Compute weighted output estimate yhat(k)
        Y = LIB1.outputEqn(ik, Xa[self.Nx+self.Nw:,:], Xx)
        yhat = np.sum(Wcx*Y, axis=-1)

        # Step 2a: Estimator gain matrix
        SigmaXY = (Xx - xhat)@np.diag(self.Wc)@(Y - yhat).T
        SigmaY  = (Y - yhat)@np.diag(self.Wc)@(Y - yhat).T
        L = SigmaXY/SigmaY

        # Step 2b: State estimate measurement update
        r = vk - yhat  # residual.  Use to check for sensor errors...
        if r**2 > 100*SigmaY: L=0
        xhat = xhat + L*r 
        xhat[0] = np.clip(xhat[0], -0.05, 1.05)
        xhat[-1] = np.clip(xhat[-1], -1, 1)

        # Step 2c: Error covariance measurement update
        SigmaX = SigmaX - L*SigmaY*L.T
        _,S,V = np.linalg.svd(SigmaX)
        HH = V*S*V.T
        SigmaX = (SigmaX + SigmaX.T + HH + HH.T)/4 # Help maintain robustness

        # Q-bump code
        if r**2>4*SigmaY: # bad voltage estimate by 2-SigmaX, bump Q 
            printf('Bumping sigmax\n')
            SigmaX[0,0] = SigmaX[0,0]*self.Qbump
        
        # Save data for next iteration...
        self.priorI = ik
        self.SigmaX = SigmaX
        self.xhat = xhat
        
        zk = self.xhat[0]
        zkbnd = 3*np.sqrt(SigmaX[0,0])
        return zk,zkbnd