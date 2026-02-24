from math import exp 
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numpy.linalg import norm
#Effective sample size Library
from arviz import ess





### Definition of the 3 mPCN Variants
### References: 
### [1] Parallel MCMC algorithms: theoretical foundations, algorithm design, case studies
###     Transactions of Mathematics and Its Applications, Volume 8, Issue 2, December 2024

    
def MpCN(q0,dim,Cov,rho,Pot,NProps,L):
    """
    The Multiproposal PcN Algorithm to samples from measure of the form
        mu(dq) = exp( - Pot(q))mu_0(dq) where mu_0 = N(0, Cov)
        The imputs are
            q0 -the initial value
            dim-dimension of the target meausre
            Cov-covariance of mu_0 (positive definate matrix)
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
        qtjCen = rho * samp[samID -1] + eta * Cov_chol @ rng.standard_normal(dim) 
        #draw initial center point
        curProps = np.concatenate((samp[samID -1][:,None],rho* qtjCen[..., None] + eta * Cov_chol @ rng.standard_normal((dim,NProps))),axis =-1).T #cloud of proposals
        logAcp = np.empty(NProps + 1, dtype=float)
        for j in range(NProps+1):
            logAcp[j] = -1*Pot(curProps[j])
        logAcp_max = np.max(logAcp)
        Acp = np.exp(logAcp - logAcp_max)  
        # stabilised weights
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
        logAcpDen_max = np.max(logAcpDenomMTM)
        #Regular Acceptance
        #MTMacpNum =exp(-1*Pot(mtmProp))*exp(-1*potpreFac* Pot(rhoinv*prevSam) -logAcpDen_max)* Acp_norm
        #MTMacpDen =exp(-1*Pot(prevSam))*exp(-1*potpreFac* Pot(rhoinv*mtmProp) -logAcp_max) *np.exp(logAcpDenomMTM - logAcpDen_max).sum()
        #MTMacp = min(1, MTMacpNum/MTMacpDen)

        
        #log acceptance
        logMTMacpNum = -1*Pot(mtmProp) -potpreFac* Pot(rhoinv*prevSam) + np.log(Acp_norm) + logAcp_max
        logMTMacpDen = -1*Pot(prevSam) - potpreFac* Pot(rhoinv*mtmProp) + np.log(np.exp(logAcpDenomMTM - logAcpDen_max).sum()) + logAcpDen_max

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
        logAcpDen_max = np.max(logAcpDenomMTM)
        #Normal Acceptance            
        #MTMacpNum = np.exp( logAcp_max- logAcpDen_max)*Acp_norm
        #MTMacpDen = np.exp(logAcpDenomMTM - logAcpDen_max).sum()
        #MTMacp = min(1, MTMacpNum/MTMacpDen)

        #Log Acceptance
        logMTMacpNum = np.log(Acp_norm) +logAcp_max
        logMTMacpDen = np.log(np.exp(logAcpDenomMTM - logAcpDen_max).sum()) + logAcpDen_max
        
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


#Functions to compute ESS per unit sample and MSJD from mpCN, mpCNBBMTM, locMpCNMTM

def msjd(samples: np.ndarray) -> float:
    """
    Computes MSJD Mean‑Squared Jumping Distances for a list of samples
    {x_0,x_1,..., x_m} as m^{-1} sum_{k=1}^{m} |x_{k-1} - x_k|^2
    The imputs are
        samples : array_like, shape (N, d)
    """
    diffs = np.diff(samples, axis=0)      # shape (N-1, d)
    return np.mean(np.sum(diffs**2, axis=1))


def autocov_fft_1d(y, L):
    n = y.size
    m = 1 << (2 * n - 1).bit_length()  # power of 2 >= 2n-1
    fy = np.fft.rfft(y, n=m)
    ac = np.fft.irfft(fy * np.conjugate(fy), n=m)[:L+1]
    denom = np.arange(n, n - (L + 1), -1, dtype=float)
    return ac / denom


##Broken
def ess_N_fixed_Lag(samples, lag):
    lsamps = len(samples)
    acovs = np.zeros(lsamps, dtype=float)
    for indx in range(lsamps):
        acovs += autocov_fft_1d(samples[indx], lag)
    acovs /= lsamps
    
    rhos = acovs / acovs[0]  # autocorrelations, rho_0 = 1
    #tau_L = 1.0 + 2.0 * float(np.sum(rhos[1:]))
    return 1/(1.0 + 2.0 * float(np.sum(rhos[1:])))


##Broken
def mixMetricsmPCNFL(q0,NumParms,Cov,rho,Pot,p,totSamps,lag):
    curSamp = MpCN(q0,NumParms,Cov,rho,Pot,p,totSamps)
    essMet =[]
    for parmIndx in range(0,NumParms):    
        essMet.append(ess_N_fixed_Lag(curSamp[:,parmIndx],lag))
    MSJDMet = msjd(curSamp)
    return essMet, MSJDMet



def mixMetricsmPCN(q0,NumParms,Cov,rho,Pot,p,totSamps):
    curSamp = MpCN(q0,NumParms,Cov,rho,Pot,p,totSamps)
    essMet =[]
    for parmIndx in range(0,NumParms):    
        essMet.append(ess(curSamp[:,parmIndx])/totSamps)
    MSJDMet = msjd(curSamp)
    return essMet, MSJDMet


def mixMetricsmPCNMulti(q0str,NumParms,Cov,rho,Pot,p,totSamps,nmRun,saveLn):
    essMet = [[] for _ in range(nmRun)]
    MSJDMet =[]
    for run in range(nmRun):
        curSamp = MpCN(q0str[run],NumParms,Cov,rho,Pot,p,totSamps)
        for parmIndx in range(0,NumParms):    
            essMet[run].append(ess(curSamp[:,parmIndx])/totSamps)
        MSJDMet.append(msjd(curSamp))
    esslst = [sum(comp) / len(comp) for comp in essMet]
    MSJD = sum(MSJDMet)/len(MSJDMet)
    return esslst, MSJD, curSamp[0:saveLn]



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

def mixMetricsmlocPCNMTMMulti(q0str,NumParms,Cov,rho,Pot,p,totSamps,nmRun,saveLn):
    essMet = [[] for _ in range(nmRun)]
    MSJDMet =[]
    for run in range(nmRun):
        curSamp = locMpCNMTM(q0str[run],NumParms,Cov,rho,Pot,p,totSamps)
        for parmIndx in range(0,NumParms): 
            curess = ess(curSamp[:,parmIndx])
            if curess == totSamps:
                essMet[run].append(0.0)
            else:
                essMet[run].append(curess/totSamps)
        MSJDMet.append(msjd(curSamp))
    esslst = [sum(comp) / len(comp) for comp in essMet]
    MSJD = sum(MSJDMet)/len(MSJDMet)
    return esslst, MSJD, curSamp[0:saveLn]

def mixMetricsBBMTM(q0,NumParms,Cov,rho,Pot,p,totSamps):
    curSamp = MpCNBBMTM(q0,NumParms,Cov,rho,Pot,p,totSamps)
    essMet =[]
    for parmIndx in range(0,NumParms):    
        essMet.append(ess_per_n_fixed_lag(curSamp[:,parmIndx])/totSamps)
    MSJDMet = msjd(curSamp)
    return essMet, MSJDMet


def mixMetricsmPCNMTMMulti(q0str,NumParms,Cov,rho,Pot,p,totSamps,nmRun,saveLn):
    essMet = [[] for _ in range(nmRun)]
    MSJDMet =[]
    for run in range(nmRun):
        curSamp = MpCNBBMTM(q0str[run],NumParms,Cov,rho,Pot,p,totSamps)
        for parmIndx in range(0,NumParms):    
            essMet[run].append(ess(curSamp[:,parmIndx])/totSamps)
        MSJDMet.append(msjd(curSamp))
    esslst = [sum(comp) / len(comp) for comp in essMet]
    MSJD = sum(MSJDMet)/len(MSJDMet)
    return esslst, MSJD, curSamp[0:saveLn]


#Potential Functions for Model Problem A

#Example Problem 1
#f(x,y) = (x-a)^p y 

def PotEx1(X,sig, a,r,z, mode = None):
    return (2* sig**2)**(-1) *(X[1]*(X[0] - a)**(r) - z)**2

#Example Problem 2
#f(x,y) = (x - x0)^2/a^2 +(y - y0)^2/b^2

def PotEx2(X, x0, y0, asqI, bsqI, sig, zdata, mode = None): 
    frwdX = asqI*(X[0] - x0)**2  + bsqI*(X[1]-y0)**2
    return (2* sig**2)**(-1) *(frwdX - zdata )**2    


#Numberical Set-up for Model Problem B




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

def PotExAD(a, gvec, sig, ModDm, z, kap, dataDim, mode = None):
    return (2*sig**2)**(-1)*(norm((z - getThA(ModDm, a, gvec, kap))[0:dataDim-1]))**2


#This is function expects data_comp to be a vector of zeros and ones.  The ones pick out the observed directions

def PotExAD_comp(a, gvec, sig, ModDm, z, kap, dataDim, data_comp, mode = None):
    return (2*sig**2)**(-1)*(norm((z - getThA(ModDm, a, gvec, kap))*data_comp ))**2


#This function expects a list of observation directions

def PotExAD_proj(Aprm, gvec, sig, ModDm, z, kap, obsdir, mode = None):
    thA = getThA(ModDm, A, gvec, kap)
    thAProj = np.array([v @ thA for v in obsdir])
    return (2*sig**2)**(-1)*(norm(z -thAProj ))**2

# Some features to search for a good value of A

def DetStiffness(dim,Pot,Cov,L):
    """
    A measure of the stiffness of a problem of the form
        mu(dq) = exp( - Pot(q))mu_0(dq) where mu_0 = N(0, Cov)
    Returns (an approximation of)
        M = \int Pot(x) \mu_0(dx),  \int (Pot(x) -M)^2 \mu_0(dx)
    """
    priorSmpLst = np.random.multivariate_normal(np.zeros(dim), Cov, size=L)
    PotValLst = [Pot(priorSmp) for priorSmp in priorSmpLst]
    return np.mean(PotValLst), np.var(PotValLst, ddof=0)


## Numerical Set-up for Problem Type C

def PotGaussPert(x, TarDim, PertDim, PriorCovInv, PostMean, PostCovInv, mode = None):
    xtrun = x[0:PertDim]
    return 1/2* xtrun.T @( PostCovInv - PriorCovInv) @ xtrun - xtrun.T@ PostCovInv @PostMean

