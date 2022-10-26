import numpy as np 
from scipy.linalg import block_diag, cholesky
from copy import copy

class SPKF:
    def __init__(self, target_model, SigmaX0, SigmaV, SigmaW):
        # store model
        self.model = copy(target_model)
        ir0   = 0                        
        hk0   = 0                        
        SOC0  = 0 # self.model.OCVfromSOC(self.model.outputEqn(ir0))
        
        self.xhat  = np.vstack([SOC0,ir0,hk0]) # initial state

        # Covariance values
        self.SigmaX = SigmaX0
        self.SigmaV = SigmaV
        self.SigmaW = SigmaW

        # SPKF specific parameters
        self.Nx = SigmaX0.shape[0]
        self.Nw = 1
        self.Nv = 1
        self.Na = self.Nx+self.Nw+self.Nv

        Wmx = np.zeros(2*self.Na+1)
        self.h = np.sqrt(3)
        Wmx[0] = (self.h**2-self.Na)/(self.h**2)
        Wmx[1:] = 1/(2*self.h**2) 
        Wcx=Wmx
        self.Wm = Wmx.reshape(-1,1) # mean
        self.Wc = Wcx.reshape(-1,1) # covar
        self.Qbump = 5

        # previous value of current
        self.priorI = 0
        
    def iter(self, ik, vk):
        # Step 1a-1: Create augmented xhat and SigmaX
        eigval, _ = np.linalg.eig(self.SigmaX)
        # print(eigval)
        sigmaXa = block_diag(self.SigmaX, self.SigmaW, self.SigmaV)
        sigmaXa = np.real(cholesky(sigmaXa, lower=True))
        xhata = np.vstack([self.xhat, np.zeros((self.Nw+self.Nv,1))])

        # Step 1a-2: Calculate SigmaX points
        Xa = xhata.reshape(self.Na,1) + self.h*np.hstack([np.zeros((self.Na, 1)), sigmaXa, -sigmaXa])

        # Step 1a-3: Time update from last iteration until now
        Xx = self.model.stateEqn(self.priorI, Xa[self.Nx:self.Nx+self.Nw,:], Xa[:self.Nx,:])
        Xx = np.vstack(Xx)
        xhat = Xx@self.Wm

        # Step 1b: Error covariance time update
        #          - Compute weighted covariance sigmaminus(k)
        #            (strange indexing of xhat to avoid "repmat" call)
        SigmaX = (Xx - xhat)@np.diag(self.Wm.ravel())@(Xx - xhat).T

        # Step 1c: Output estimate
        #          - Compute weighted output estimate yhat(k)
        Y = self.model.outputEqn(ik, Xa[self.Nx+self.Nw:,:], Xx)
        yhat = Y@self.Wc

        # Step 2a: Estimator gain matrix
        SigmaXY = (Xx - xhat)@np.diag(self.Wc.ravel())@(Y - yhat).T
        SigmaY  = (Y - yhat)@np.diag(self.Wc.ravel())@(Y - yhat).T
        L = SigmaXY/SigmaY

        # Step 2b: State estimate measurement update
        r = vk - yhat  # residual.  Use to check for sensor errors...
        if r**2 > 100*SigmaY: L=0
        # print(L, r, yhat, vk)
        xhat = xhat + L*r 
        xhat[0] = np.clip(xhat[0], -0.05, 1.05)
        xhat[-1] = np.clip(xhat[-1], -1, 1)

        # Step 2c: Error covariance measurement update
        SigmaX = SigmaX - L*SigmaY*L.T
        _,S,V = np.linalg.svd(SigmaX)
        HH = V@np.diag(S)@V.T
        SigmaX = (SigmaX + SigmaX.T + HH + HH.T)/4 # Help maintain robustness
        
        # Q-bump code
        if r**2>4*SigmaY: # bad voltage estimate by 2-SigmaX, bump Q 
            print('Bumping sigmax\n')
            SigmaX[0,0] = SigmaX[0,0]*self.Qbump
        
        # Save data for next iteration...
        self.priorI = ik
        self.SigmaX = SigmaX
        self.xhat = xhat
        
        zk = self.xhat[0]
        zkbnd = 3*np.sqrt(SigmaX[0,0])
        return zk,zkbnd