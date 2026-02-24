from math import exp 
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numpy.linalg import norm


def MpCN(q0,dim,Cov,rho,Pot,NProps,L):
    """
    The Multiproposal PcN Algorithm 
        Provdies to samples from measure of the form
        mu(dq) = exp( - Pot(q))mu_0(dq) where mu_0 = N(0, Cov)
        The imputs are
            q0 -initial value
            dim-dimension of the target meausre
            Cov-covariance of mu_0
            rho - algorithmic parameter taking values in [0,1)
            Pot- potential `loglikelihood' term in mu
            NProps-number of proposals per step ('p')
            L-total number of iteration steps
    """
    rng = np.random.default_rng()
    samp = np.empty((L + 1, dim), dtype=float) #Make an array for the samples
    samp[0] = np.array(q0)
    Cov_chol = np.linalg.cholesky(Cov) #Find E such that EE^* = C
    eta = np.sqrt(1.0 - rho * rho)
    for samID in range(1, L + 1):
        qtjCen = rho * samp[samID -1] + eta * Cov_chol @ rng.standard_normal(dim) #draw initial center point
        curProps = np.concatenate((samp[samID -1][:,None],rho* qtjCen[..., None] + eta * Cov_chol @ rng.standard_normal((dim,NProps))),axis =-1).T #cloud of proposals
        logAcp = np.empty(NProps + 1, dtype=float)
        for j in range(NProps+1):
            logAcp[j] = -1*Pot(curProps[j])
        logAcp_max = np.max(logAcp)
        Acp = np.exp(logAcp - logAcp_max)  # stabilised weights
        idx = rng.choice(NProps+1, p=Acp / Acp.sum())
        samp[samID] = curProps[idx].copy()
    return samp 




def locMpCNMTM(q0,dim,Cov,rho,Pot,NProps,L,PrintAcpRate = False):
    """
    The `local Multiproposal PcN Algorithm' with the MTM correction
        Samples from measure of the form
        mu(dq) = exp( - Pot(q))mu_0(dq) where mu_0 = N(0, Cov)
        The imputs are
            q0 -initial value
            dim-dimension of the target meausre
            Cov-covariance of mu_0
            rho - algorithmic parameter taking values in [0,1)
            Pot- potential `loglikelihood' term in mu
            NProps-number of proposals per step ('p')
            L-total number of iteration steps
    """
    nmRjt = 0.0
    rng = np.random.default_rng()
    samp = np.empty((L + 1, dim), dtype=float) #Make an array for the samples
    samp[0] = np.array(q0)
    Cov_chol = np.linalg.cholesky(Cov) 
    #Find E such that EE^* = C
    eta = np.sqrt(1.0 - rho * rho)
    #Scaling prefactor in the potential
    potpreFac = rho*(1.0 - rho)*(1.0 - rho * rho)**(-1) 
    rhoinv = rho**(-1)
    for samID in range(1, L + 1):
        prevSam = samp[samID -1].copy()
        curProps = (rho* prevSam[:,None]  + eta * Cov_chol @ rng.standard_normal((dim,NProps))).T 
        #the main cloud of proposals
        logAcp = np.empty(NProps, dtype=float)
        #log acceptance probabilities
        for j in range(NProps):
            logAcp[j] = -1* potpreFac*Pot(rhoinv*curProps[j])
        logAcp_max = np.max(logAcp)
        Acp = np.exp(logAcp - logAcp_max)  # stabilised weights
        Acp_norm = Acp.sum()
        idx = rng.choice(NProps, p=Acp / Acp_norm)
        mtmProp = curProps[idx].copy()

        #MTM Backward Step
        
        mtmPropsAux = (rho* mtmProp[:,None] + eta * Cov_chol @ rng.standard_normal((dim,NProps -1))).T 
        #MTM Reverse Proproposal
        logAcpDenomMTM = np.empty(NProps, dtype=float)
        logAcpDenomMTM[0] = -1*potpreFac*Pot(rhoinv*prevSam)
        for j in range(NProps -1):
            logAcpDenomMTM[j+1] = -1* potpreFac*Pot(rhoinv*mtmPropsAux[j])
        #logAcpDen_max = np.max(logAcpDenomMTM)
        #Regular Acceptance
        #MTMacpNum =exp(-1*Pot(mtmProp))*exp(-1*potpreFac* Pot(rhoinv*prevSam) -logAcpDen_max)* Acp_norm
        #MTMacpDen =exp(-1*Pot(prevSam))*exp(-1*potpreFac* Pot(rhoinv*mtmProp) -logAcp_max) *np.exp(logAcpDenomMTM - logAcpDen_max).sum()
        #MTMacp = min(1, MTMacpNum/MTMacpDen)

        #log acceptance
        logMTMacpNum = -1*Pot(mtmProp) -potpreFac* Pot(rhoinv*prevSam) + logAcp.sum()
        logMTMacpDen = -1*Pot(prevSam) - potpreFac* Pot(rhoinv*mtmProp) + logAcpDenomMTM.sum() 

        logMTMacp = min(0, logMTMacpNum -logMTMacpDen)
        MTMacp = np.exp(logMTMacp)
        
        if rng.random() < MTMacp:
            samp[samID] = mtmProp
        else:
            nmRjt += 1
            samp[samID] = prevSam

    if PrintAcpRate:
        print("Rejection rate:")
        print(nmRjt/L)
        
    return samp

def MpCNBBMTM(q0,dim,Cov,rho,Pot,NProps,L,PrintAcpRate = False):
    """
    The `bubble bath' Multiproposal PcN Algorithm with MTM correction.
        Samples from measure of the form
        mu(dq) = exp( - Pot(q))mu_0(dq) where mu_0 = N(0, Cov)
        The imputs are
            q0 -initial value
            dim-dimension of the target meausre
            Cov-covariance of mu_0
            rho - algorithmic parameter taking values in [0,1)
            Pot- potential `loglikelihood' term in mu
            NProps-number of proposals per step ('p')
            L-total number of iteration steps
    """
    nmRjt = 0.0
    rng = np.random.default_rng()
    samp = np.empty((L + 1, dim), dtype=float) #Make an array for the samples
    samp[0] = np.array(q0)
    Cov_chol = np.linalg.cholesky(Cov) 
    #Find E such that EE^* = C
    eta = np.sqrt(1.0 - rho * rho)
    for samID in range(1, L + 1):
        curProps = (rho* samp[samID -1][:,None]  + eta * Cov_chol @ rng.standard_normal((dim,NProps))).T 
        #the main cloud of proposals
        logAcp = np.empty(NProps, dtype=float)
        #log acceptance probabilities
        for j in range(NProps):
            logAcp[j] = -1* Pot(curProps[j])
        logAcp_max = np.max(logAcp)
        Acp = np.exp(logAcp - logAcp_max)  # stabilised weights
        Acp_norm = Acp.sum()
        idx = rng.choice(NProps, p=Acp / Acp_norm)
        mtmProp = curProps[idx].copy()

        #MTM Backward Step
        
        mtmPropsAux = (rho* mtmProp[:,None] + eta * Cov_chol @ rng.standard_normal((dim,NProps -1))).T 
        #MTM Reverse Proproposal
        logAcpDenomMTM = np.empty(NProps, dtype=float)
        prevSam = samp[samID -1].copy()
        logAcpDenomMTM[0] = -1*Pot(prevSam)
        for j in range(NProps -1):
            logAcpDenomMTM[j+1] = -1* Pot(mtmPropsAux[j])

        #Normal Acceptance            
        #logAcpDen_max = np.max(logAcpDenomMTM)
        #MTMacpNum = np.exp( logAcp_max- logAcpDen_max)*Acp_norm
        #MTMacpDen = np.exp(logAcpDenomMTM - logAcpDen_max).sum()
        #MTMacp = min(1, MTMacpNum/MTMacpDen)

        #Log Acceptance
        logMTMacpNum = logAcp.sum()
        logMTMacpDen = logAcpDenomMTM.sum()
        logMTMacp = min(0, logMTMacpNum -logMTMacpDen)
        MTMacp = np.exp(logMTMacp)
        
        if rng.random() < MTMacp:
            samp[samID] = mtmProp
        else:
            nmRjt += 1
            samp[samID] = prevSam
    if PrintAcpRate:
        print("Rejection rate:")
        print(nmRjt/L)
    return samp


#Effective sample size Library
from arviz import ess


def mixMetricsmPCN(q0,NumParms,Cov,rho,Pot,p,totSamps):
    curSamp = MpCN(q0,NumParms,Cov,rho,Pot,p,totSamps)
    essMet =[]
    for parmIndx in range(0,NumParms):    
        essMet.append(ess(curSamp[:,parmIndx])/totSamps)
    MSJDMet = msjd(curSamp)
    return essMet, MSJDMet

def mixMetricsmlocPCNMTM(q0,NumParms,Cov,rho,Pot,p,totSamps):
    curSamp = locMpCNMTM(q0,NumParms,Cov,rho,Pot,p,totSamps)
    essMet =[]
    for parmIndx in range(0,NumParms):
        curess = ess(curSamp[:,parmIndx])
        if curess == totSamps:
            essMet.append(0.0)
        else:
            essMet.append(curess/totSamps)
    MSJDMet = msjd(curSamp)
    return essMet, MSJDMet

def mixMetricsBBMTM(q0,NumParms,Cov,rho,Pot,p,totSamps):
    curSamp = MpCNBBMTM(q0,NumParms,Cov,rho,Pot,p,totSamps)
    essMet =[]
    for parmIndx in range(0,NumParms):    
        essMet.append(ess(curSamp[:,parmIndx])/totSamps)
    MSJDMet = msjd(curSamp)
    return essMet, MSJDMet





#To compute some mixing measures

def msjd(samples: np.ndarray) -> float:
    """
    Mean‑Squared Jumping Distance.

    Parameters
    ----------
    samples : array_like, shape (N, d)
        Markov‑chain states X₁,…,X_N.

    Returns
    -------
    msjd : float
        Average squared Euclidean distance between successive states.
    """
    diffs = np.diff(samples, axis=0)      # shape (N-1, d)
    return np.mean(np.sum(diffs**2, axis=1))


#Example Problem 1
#We consider the inverse problem 
#y = (x1-a)^p x2 + \eta  where \eta \sim N(0,\sigma^2)

#Specification of the forward problem (MM)
#y = 6
#a = .3
#p = 1
#sig = 1
#fFn = lambda X : (X[0] - a)**(p)*X[1]
#fFnStr = "(x1-" + str(a) + ")^" + str(p) + "x2"
#NumParmsEx2 = 2
#LogLikihood function
#PotEx2 = lambda X : (2* sig**2)**(-1) *(fFn(X) - y )**2

#Reset the defaults to change parameters
def PotEx1(X,sig =1, a =.3,r=1,y=6):
    return (2* sig**2)**(-1) *((X[0] - a)**(r)*X[1] - y )**2


#Specifying the covariance
#CovEVs = [3,2]
#Rot = np.array([
#        [np.cos(Tth), -np.sin(Tth)],
#        [np.sin(Tth),  np.cos(Tth)]
#    ])
#Diag = np.diag(CovEVs)
#CovEx2 = Rot.T @ Diag @ Rot
#CovEx2 = Diag




#Example Problem 2

#Numerical Set up for AD Toy Model
#AD Toy Model

def MkAD_A_Mat(ModDim, curApar):
    """
    Generates an antisymmetric matric with the given Model Parameters
    """
    A = np.zeros([ModDim,ModDim])
    A[np.triu_indices(ModDim, k=1)] = curApar 
    #triu_indices returns the indices of all the above diagonal indicies
    return A - A.T

def getThA(ModDim, Apar, g, kappa):
    """
    Solves (A + kI) th = g  for theta given the specified model parameters determining A
    """
    A_p_kI = MkAD_A_Mat(ModDim, Apar)+ kappa*np.identity(ModDim)
    return np.linalg.solve(A_p_kI,g)

def mkDiagCov(vrs):
    return np.diag(vrs)


#Code to generate a random orthogonal matrix (uniform from O(n)) using the QR factorization of a Guassian matrix.
def rndm_orth_matrix(n):

    #Generate a random n x n matrix with i.i.d. normal entries
    A = np.random.randn(n, n)
    
    #Perform the QR factorization
    Q, R = np.linalg.qr(A)    
    
    return Q



#Specifying Problem Parameters
#Model Dimension and Parameter Size

ModDm = 4
NumParmsAD = int(ModDm*(ModDm -1)/2)

# Specify Potential and Prior Covariance
# The Posterior is of the form
# mu(dA) = Z^{-1} \exp( -1/(2 sig^2) ( (y0 - th(A)(0))^2 +(y1 - th(A)(0))^2 ) mu_0(dA)
# where
# mu_0(dA) = Z^{-1}_0 \exp( - 1/2<C^{-1}A, A>)
# Here The Forward Model entails solving for th(A) for any antisymmetric A where
# (A + kap I) th = g 
# so that th(A) = th_{k,g}
# g = (g0,g1, g2, g3)^T


g =  [.1,0,5,2]
gvec = np.transpose(np.array(g))
sig = 2
ModDm =4 
y = np.array([4.601,18.021])
kap= .05

#PotEx2 = lambda a : (2*sig**2)**(-1)*(norm(y - getThA(ModDm, a, g, kap)[0:2]))**2
#def PotEx2(a, sig = 2,ModDm =4, y = [4.601,18.021,0,0], g =  [.1,0,5,2], kap= kap):

def PotEx2(a):
    return (2*sig**2)**(-1)*(norm(y - getThA(ModDm, a, gvec, kap)[0:2]))**2
# Covariance of the 'prior' C = cov0[1^{-gam}, 2^[-gam],..., N^{-gam}]
# where N is the number of parameters in the model NumParms = 6 
#for number of enetries that need to be specified in A
#cov0 = 5
#gam = 1.5
#CovDiag = [cov0* (j**(-gam)) for j in list(range(1,NumParmsAD+1))]
#CovAD = mkDiagCov(CovDiag)

#Example 3: Statistical Inversion for f(x,y) = (x - x_0)^2/a^2 + (y-y_0)^2/b^2

#Specification of the forward problem (MM)
zdata = 2
sig = .05
a0 = 1
b0 = 3
asqI = a0**(-2) 
bsqI = b0 **(-2)
x0 = 2
y0 = 2

def PotEx3(X): 
    frwdX = asqI*(X[0] - x0)**2  + bsqI*(X[1]-y0)**2
    return (2* sig**2)**(-1) *(frwdX - zdata )**2    
    
fFnStr = "(x - " + str(x0) + ")^2*" + str(a0) + "^(-2) + (y - " + str(x0) + ")^2*" + str(b0) + "^(-2)"

#Example 4: Stiff AD problem

ModDmEx4 = 4
NumParmsEx4 = int(ModDmEx4*(ModDmEx4 -1)/2)

gvecEx4 = np.ones(ModDmEx4)
sigEx4 = 0.5
kapEx4 = .2
yEx4 = getThA(ModDmEx4, np.zeros(NumParmsEx4), gvecEx4, kapEx4)


siginvSq = (2*sigEx4**2)**(-1)

def PotEx4(A):
    return siginvSq * (norm(yEx4 - getThA(ModDmEx4, A, gvecEx4, kapEx4)))**2

