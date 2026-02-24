from math import exp 
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numpy.linalg import norm
#Effective sample size Library
from arviz import ess
import arviz as az
import xarray as xr



#Numerical Elements
from numpy.linalg import norm
import numpy as np
from numpy import dot, array, transpose, diag
import random
import math

#Input Output utils
import os
import pandas as pd

#Stats elements
from scipy.stats import norm

#Plotting stuff
from matplotlib.patches import Ellipse

#Finished Message
import smtplib
from email.message import EmailMessage



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



def mixMetricsmPCN(q0_stat,NumParms,Cov,Pot,rho,p,ESSlen,ESSruns,Nave,numRuns,numMom,saveLn):
    r"""
    -q0_stat: is intended as an i.i.d list of initial condition from pi of length numRuns + ESSruns
    -NumParms: Dimenion of target
    -Cov: Covariance of the prior
    -Pot: Potential of target exp( - Pot(x)) mu_0(dx)
    -rho: size of pCN steps
    -p: number of elements of proposal cloud
    -ESSlen: Length of chain to compute ESS/MSJD 
    -ESSruns: Number of independent ESS/MSJD chains
    -Nave: size of runs N for bar{g}_N^\rho = 1/N \sum_{k =1}^N
    -numRuns: number of independent samples from mu_0
    -numMom: Number of moments to compute for the observables
    -saveLn: How long to save the chain to give chain time series diagonistics
    """
    MSJDdiff_sum = 0.0
    MSJDdiff_count = 0
    ESS_Runs = []
    for parmIndx in range(0,ESSruns):
        curSamp = MpCN(q0_stat[parmIndx],NumParms,Cov,rho,Pot,p,ESSlen)
        dc = np.diff(curSamp, axis=0)      # shape: (Nave, NumParms) if curSamp is (Nave+1, NumParms)
        MSJDdiff_sum += np.sum(dc * dc)
        ESS_Runs.append(curSamp)
    chainStack = np.stack(ESS_Runs, axis=0)     # (chain, draw, d)
    n_chain, n_draw, d = chainStack.shape
    # --- ESS per component using xarray (this fixes your ArviZ error) ---
    da = xr.DataArray(
        chainStack,
        dims=("chain", "draw", "dim"),
        coords={"chain": np.arange(n_chain),
                "draw": np.arange(n_draw),
                "dim": np.arange(d)},
        name="x",
    )
    ess_da = az.ess(da)                               # DataArray over dim
    ess_vec = ess_da["x"].values                           # numpy array length d
    MSJD = MSJDdiff_sum/((ESSlen-1)*ESSruns)

    gNmeans = np.empty((numRuns, NumParms), dtype=float)          # each run's mean vector
    gNCov   = np.empty((numRuns, NumParms, NumParms), dtype=float)       # each run's sample cov matrix
    gNMom   = np.empty((numRuns, numMom), dtype=float)          # each run's moments (norm^j)
    #gNWTN   = np.empty(numRuns, dtype=float)               # each run's avg x^T C^{-1} x ###PROBABLY A BAD IDEA

    for parmIndx in range(0,numRuns):
        curSampgN = MpCN(q0_stat[ESSruns+parmIndx],NumParms,Cov,rho,Pot,p,Nave)
        gNmeans[parmIndx] = curSampgN.mean(axis=0)
        gNCov[parmIndx] = np.cov(curSampgN, rowvar=False, bias=False)
        gNnorms = np.linalg.norm(curSampgN, axis=1)
        for j in range(1, numMom + 1):
            gNMom[parmIndx, j-1] = np.mean(gNnorms**j)
        #cCurSampgN = np.linalg.solve(Cov, curSampgN.T).T
        #gNWTN[parmIndx] = np.einsum('ni,ij,nj->', curSampgN, cCurSampgN) / Nave
        
    var_gNmean = gNmeans.var(axis=0, ddof=1)
    var_gNCov_entries = gNCov.var(axis=0, ddof=1) 
    var_gNMom = gNMom.var(axis=0, ddof=1)

    means_gNmean = gNmeans.mean(axis=0)
    means_gNCov_entries = gNCov.mean(axis=0) 
    means_gNMom = gNMom.mean(axis=0)

    #var_gNWTN = gNWTN.var(ddof=1) 

    # return a dictionary of with run parameters + 
    # ESS, MSJD, g_N = 1/N \sum_{j =1}^N g(X_j) for various g, and some samples 
    my_big_dict = {}
    my_big_dict["ESS_Smp_Size"] = ESSlen
    my_big_dict["ESS_Ind_Runs"] = ESSruns
    my_big_dict["Estimator_Ln"] = Nave
    my_big_dict["num_Estimator_runs"] = numRuns
    my_big_dict["ESS"] = ess_vec
    my_big_dict["MSJD"] = MSJD
    my_big_dict["Var(gNmean)"] = var_gNmean
    my_big_dict["Var(gNCov)"] = var_gNCov_entries
    my_big_dict["Var(gNMoms)"] = var_gNMom
    my_big_dict["mean(gNmean)"] = means_gNmean
    my_big_dict["mean(gNCov)"] = means_gNCov_entries
    my_big_dict["mean(gNMoms)"] = means_gNMom
    my_big_dict["num Moms Saved"] = numMom
    #my_big_dict["var(gNWtnMean)"] = var_gNWTN
    my_big_dict["time_series"] = curSamp[0:saveLn]
    return my_big_dict 





def mixMetricsmlocPCNMTM(q0_stat,NumParms,Cov,Pot,rho,p,ESSlen,ESSruns,Nave,numRuns,numMom,saveLn):
    r"""
    -q0_stat: is intended as an i.i.d list of initial condition from pi of length numRuns + ESSruns
    -NumParms: Dimenion of target
    -Cov: Covariance of the prior
    -Pot: Potential of target exp( - Pot(x)) mu_0(dx)
    -rho: size of pCN steps
    -p: number of elements of proposal cloud
    -ESSlen: Length of chain to compute ESS/MSJD 
    -ESSruns: Number of independent ESS/MSJD chains
    -Nave: size of runs N for bar{g}_N^\rho = 1/N \sum_{k =1}^N
    -numRuns: number of independent samples from mu_0
    -numMom: Number of moments to compute for the observables
    -saveLn: How long to save the chain to give chain time series diagonistics
    """
    MSJDdiff_sum = 0.0
    MSJDdiff_count = 0
    ESS_Runs = []
    for parmIndx in range(0,ESSruns):
        curSamp = locMpCNMTM(q0_stat[parmIndx],NumParms,Cov,rho,Pot,p,ESSlen)
        dc = np.diff(curSamp, axis=0)      # shape: (Nave, NumParms) if curSamp is (Nave+1, NumParms)
        MSJDdiff_sum += np.sum(dc * dc)
        ESS_Runs.append(curSamp)
    chainStack = np.stack(ESS_Runs, axis=0)     # (chain, draw, d)
    n_chain, n_draw, d = chainStack.shape
    # --- ESS per component using xarray (this fixes your ArviZ error) ---
    da = xr.DataArray(
        chainStack,
        dims=("chain", "draw", "dim"),
        coords={"chain": np.arange(n_chain),
                "draw": np.arange(n_draw),
                "dim": np.arange(d)},
        name="x",
    )
    ess_da = az.ess(da)                               # DataArray over dim
    ess_vec = ess_da["x"].values                           # numpy array length d
    MSJD = MSJDdiff_sum/((ESSlen-1)*ESSruns)

    gNmeans = np.empty((numRuns, NumParms), dtype=float)          # each run's mean vector
    gNCov   = np.empty((numRuns, NumParms, NumParms), dtype=float)       # each run's sample cov matrix
    gNMom   = np.empty((numRuns, numMom), dtype=float)          # each run's moments (norm^j)
    #gNWTN   = np.empty(numRuns, dtype=float)               # each run's avg x^T C^{-1} x ###PROBABLY A BAD IDEA

    for parmIndx in range(0,numRuns):
        curSampgN = locMpCNMTM(q0_stat[ESSruns+parmIndx],NumParms,Cov,rho,Pot,p,Nave)
        gNmeans[parmIndx] = curSampgN.mean(axis=0)
        gNCov[parmIndx] = np.cov(curSampgN, rowvar=False, bias=False)
        gNnorms = np.linalg.norm(curSampgN, axis=1)
        for j in range(1, numMom + 1):
            gNMom[parmIndx, j-1] = np.mean(gNnorms**j)
        #cCurSampgN = np.linalg.solve(Cov, curSampgN.T).T
        #gNWTN[parmIndx] = np.einsum('ni,ij,nj->', curSampgN, cCurSampgN) / Nave
        
    var_gNmean = gNmeans.var(axis=0, ddof=1)
    var_gNCov_entries = gNCov.var(axis=0, ddof=1) 
    var_gNMom = gNMom.var(axis=0, ddof=1)
    #var_gNWTN = gNWTN.var(ddof=1) 

    # return a dictionary of with run parameters + 
    # ESS, MSJD, g_N = 1/N \sum_{j =1}^N g(X_j) for various g, and some samples 
    my_big_dict = {}
    my_big_dict["ESS_Smp_Size"] = ESSlen
    my_big_dict["ESS_Ind_Runs"] = ESSruns
    my_big_dict["Estimator_Ln"] = Nave
    my_big_dict["num_Estimator_runs"] = numRuns
    my_big_dict["ESS"] = ess_vec
    my_big_dict["MSJD"] = MSJD
    my_big_dict["Var(gNmean)"] = var_gNmean
    my_big_dict["Var(gNCov)"] = var_gNCov_entries
    my_big_dict["Var(gNMoms)"] = var_gNMom
    #my_big_dict["var(gNWtnMean)"] = var_gNWTN
    my_big_dict["time_series"] = curSamp[0:saveLn]
    return my_big_dict 






def mixMetricsBBMTM(q0_stat,NumParms,Cov,Pot,rho,p,ESSlen,ESSruns,Nave,numRuns,numMom,saveLn):
    r"""
    -q0_stat: is intended as an i.i.d list of initial condition from pi of length numRuns + ESSruns
    -NumParms: Dimenion of target
    -Cov: Covariance of the prior
    -Pot: Potential of target exp( - Pot(x)) mu_0(dx)
    -rho: size of pCN steps
    -p: number of elements of proposal cloud
    -ESSlen: Length of chain to compute ESS/MSJD 
    -ESSruns: Number of independent ESS/MSJD chains
    -Nave: size of runs N for bar{g}_N^\rho = 1/N \sum_{k =1}^N
    -numRuns: number of independent samples from mu_0
    -numMom: Number of moments to compute for the observables
    -saveLn: How long to save the chain to give chain time series diagonistics
    """
    MSJDdiff_sum = 0.0
    MSJDdiff_count = 0
    ESS_Runs = []
    for parmIndx in range(0,ESSruns):
        curSamp = MpCNBBMTM(q0_stat[parmIndx],NumParms,Cov,rho,Pot,p,ESSlen)
        dc = np.diff(curSamp, axis=0)      # shape: (Nave, NumParms) if curSamp is (Nave+1, NumParms)
        MSJDdiff_sum += np.sum(dc * dc)
        ESS_Runs.append(curSamp)
    chainStack = np.stack(ESS_Runs, axis=0)     # (chain, draw, d)
    n_chain, n_draw, d = chainStack.shape
    # --- ESS per component using xarray (this fixes your ArviZ error) ---
    da = xr.DataArray(
        chainStack,
        dims=("chain", "draw", "dim"),
        coords={"chain": np.arange(n_chain),
                "draw": np.arange(n_draw),
                "dim": np.arange(d)},
        name="x",
    )
    ess_da = az.ess(da)                               # DataArray over dim
    ess_vec = ess_da["x"].values                           # numpy array length d
    MSJD = MSJDdiff_sum/((ESSlen-1)*ESSruns)

    gNmeans = np.empty((numRuns, NumParms), dtype=float)          # each run's mean vector
    gNCov   = np.empty((numRuns, NumParms, NumParms), dtype=float)       # each run's sample cov matrix
    gNMom   = np.empty((numRuns, numMom), dtype=float)          # each run's moments (norm^j)
    #gNWTN   = np.empty(numRuns, dtype=float)               # each run's avg x^T C^{-1} x ###PROBABLY A BAD IDEA

    for parmIndx in range(0,numRuns):
        curSampgN = MpCNBBMTM(q0_stat[ESSruns+parmIndx],NumParms,Cov,rho,Pot,p,Nave)
        gNmeans[parmIndx] = curSampgN.mean(axis=0)
        gNCov[parmIndx] = np.cov(curSampgN, rowvar=False, bias=False)
        gNnorms = np.linalg.norm(curSampgN, axis=1)
        for j in range(1, numMom + 1):
            gNMom[parmIndx, j-1] = np.mean(gNnorms**j)
        #cCurSampgN = np.linalg.solve(Cov, curSampgN.T).T
        #gNWTN[parmIndx] = np.einsum('ni,ij,nj->', curSampgN, cCurSampgN) / Nave
        
    var_gNmean = gNmeans.var(axis=0, ddof=1)
    var_gNCov_entries = gNCov.var(axis=0, ddof=1) 
    var_gNMom = gNMom.var(axis=0, ddof=1)
    #var_gNWTN = gNWTN.var(ddof=1) 

    # return a dictionary of with run parameters + 
    # ESS, MSJD, g_N = 1/N \sum_{j =1}^N g(X_j) for various g, and some samples 
    my_big_dict = {}
    my_big_dict["ESS_Smp_Size"] = ESSlen
    my_big_dict["ESS_Ind_Runs"] = ESSruns
    my_big_dict["Estimator_Ln"] = Nave
    my_big_dict["num_Estimator_runs"] = numRuns
    my_big_dict["ESS"] = ess_vec
    my_big_dict["MSJD"] = MSJD
    my_big_dict["Var(gNmean)"] = var_gNmean
    my_big_dict["Var(gNCov)"] = var_gNCov_entries
    my_big_dict["Var(gNMoms)"] = var_gNMom
    #my_big_dict["var(gNWtnMean)"] = var_gNWTN
    my_big_dict["time_series"] = curSamp[0:saveLn]
    return my_big_dict 


def message_me(subject,message):
    user = "nglattholtz@gmail.com"
    pw = "irvl thcq apqa fnvw"
    to = "8187235144@vtext.com"
    host = "smtp.gmail.com"
    port = "465"


    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(message)


    with smtplib.SMTP_SSL(host, port) as smtp:
        smtp.login(user, pw)
        smtp.send_message(msg)


def writeCSV(filenm, sampArray, newFile = False):
    if newFile:
        pd.DataFrame(sampArray).to_csv(filenm, mode='w', index=False, header=False)
    else:
        pd.DataFrame(sampArray).to_csv(filenm, mode='a', index=False, header=False)

def readCSV(filenm):   
    df = pd.read_csv(filenm)
    MpCNsamp2 = df.to_numpy()
    return MpCNsamp2

#Fun Progress Bar
from tqdm.notebook import tqdm

#Parallel executation functions
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import inspect
#print("Current cpu count:" + str(mp.cpu_count()))

#Histogram and Other Graphics Utililities
import matplotlib.pyplot as plt
import matplotlib as mpl

def getComp(sampLst,indx):
    return [item[indx] for item in sampLst]

def makeHistGrid(R, dr, sampList, NumParams,saveLoc, C=None, beta=0.95, hidePlt = True):
    SampLstsGd = [getComp(sampList,j) for j in range(0,NumParams)]
    numbins= int(2*R/dr)
    x_bins = np.linspace(-R, R, numbins) 
    y_bins = np.linspace(-R, R, numbins)

    fig, axs = plt.subplots(NumParams, NumParams,figsize=(15,15))
    for i in range(0,NumParams):
        for j in range(0,NumParams):
            if i == j:
                axs[i,j].hist(SampLstsGd[i], density=True, bins=x_bins)
                # Add confidence interval ticks if covariance provided
                if C is not None:
                    # Quantile of standard normal
                    z = norm.ppf((1 + beta) / 2)
                    # Confidence interval endpoints
                    bound = z * np.sqrt(C[i, i])
                    # Get current y-axis limits for tick height
                    ylim = axs[i, j].get_ylim()
                    tick_height = 0.08 * (ylim[1] - ylim[0])
                    # Draw ticks at -bound and +bound
                    for x_pos in [-bound, bound]:
                        axs[i, j].axvline(x=x_pos, ymin=0, ymax=0.08, color='red', linewidth=2.5)   
                    # Add a small label
                    #axs[i, j].text(0, ylim[1] * 0.92, f'{100*beta:.0f}% CI', ha='center', va='top', fontsize=8, color='red')
            else:
                axs[i,j].hist2d(SampLstsGd[j],SampLstsGd[i],bins= [x_bins,y_bins])
                if C is not None:
                    # Extract 2x2 marginal covariance (note: hist2d has j on x-axis, i on y-axis)
                    Sigma = np.array([[C[j, j], C[j, i]],[C[i, j], C[i, i]]])
                    # Chi-squared quantile with 2 degrees of freedom
                    # For chi^2_2: quantile = -2 * ln(1 - beta)
                    chi2_val = -2 * np.log(1 - beta)
                    # Eigendecomposition of Sigma
                    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
                    # Sort by eigenvalue (largest first) for consistent orientation
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
    
                    # Semi-axis lengths: sqrt(eigenvalue * chi2_quantile)
                    width = 2 * np.sqrt(eigenvalues[0] * chi2_val)
                    height = 2 * np.sqrt(eigenvalues[1] * chi2_val)
                    # Rotation angle (in degrees) from the first eigenvector
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
                    # Create and add the ellipse patch
                    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                                      edgecolor='red', facecolor='none', linewidth=2, 
                                      linestyle='-', zorder=10)
                    axs[i,j].add_patch(ellipse)
        for ax in axs.flat:
            ax.label_outer()
    plt.savefig(saveLoc)
    if hidePlt:
        plt.close()


def plot_timeseries(samples, filename, MCMC_type, burn_in=0):
    r"""
    Plot time series (trace plots) for each component of the MCMC chain
    and save to a file.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (L+1, dim) returned by MpCN. Each row is an iteration,
        each column is a component.
    filename : str
        Path (including extension, e.g. 'traceplots.png' or 'traceplots.pdf')
        where the figure will be saved.
    burn_in : int, optional
        Number of initial samples to discard before plotting.
    """
    # Ensure numpy array
    samples = np.asarray(samples)
    n_iter, dim = samples.shape

    # Indices of iterations after burn-in
    iters = np.arange(burn_in, n_iter)

    # Make one subplot per dimension
    fig, axes = plt.subplots(dim, 1, figsize=(8, 2.0 * dim), sharex=True)

    # If dim == 1, axes is not a list
    if dim == 1:
        axes = [axes]

    for d in range(dim):
        axes[d].plot(iters, samples[burn_in:, d])
        axes[d].set_ylabel(f"$q_{d+1}$")
        axes[d].grid(alpha=0.3)

    axes[-1].set_xlabel("Iteration")

    fig.suptitle(MCMC_type + " Trace Plots", y=0.99)
    fig.tight_layout()

    # Ensure directory exists (if any directory is specified)
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save and close
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)




def generate_colors(n, cmap_name="tab10"):
    r"""
    Return a list of `n` hex colors taken at equal intervals
    from a Matplotlib colormap (default: 'tab10').

    Parameters
    ----------
    n : int
        Number of colors you need.
    cmap_name : str, optional
        Any valid Matplotlib colormap name.  Sequential maps
        ('viridis', 'plasma', …) or qualitative maps ('tab10',
        'Set3', …) both work.

    Returns
    -------
    list[str]
        Hex strings like '#1f77b4' ready for plotting.
    """
    # Get a ListedColormap with exactly n entries
    cmap = plt.get_cmap(cmap_name, n)
    # Convert RGBA to hex for convenience
    return [mpl.colors.to_hex(cmap(i)) for i in range(cmap.N)]



def parameter_sweep_p_rho(ImpLst, q0FN, TargetDim, CovPrior,Pot, SampleLnSv = 2000, NSampsingN = 100, MomLen = 4, saveDictExt = False, saveDictLoc = False):
    """
        ImpList
        FORMAT: [p,NumRho,NumSampsESS,numChainsESS, NumChainsgM, q0zSt]
            -p: value to p to run study
            -NumRho: 1/NumRho specfies the step size in rho over [0,1] for the study
            -NumSampsESS: Length of the MCMC at each rho value to compute ESS/MSJD
            -numChainsESS: Number of independent chains to compute ESS/MSJD
            -NumChainsgM: Number of separate chain M= NumChainsgM to compute Var(\bar{g}_N^\rho)

        Input to studies 
        Across p parameters

        SampleLnSv = 2000 #Number of samples to save for time series and other diagonistics
        NSampsingN = 100 #Number of samples N = NumSampsgN in bar{g}_N^rho = 1/N sum_{j =1}^N g(X_j^\rho)
        MomLen = 4 #Number of moments to compute per study for bar{g}_N^rho

        q0zSt --Randomly sampled initial list from target mu
             Should have dimensions NumChainsESS+NumChainsgM
    """
    
    sample_run_dict ={} #Make an empty dictionary to store sample runs
    prho_study_data_dict ={} #Make an empty dictionary to store all outputs of p/rho sweep

    prho_study_data_dict["Input List"] = ImpLst #save input list
    prhoList = []
    algLst = ["mpCNOG","mpCNMTMLoc", "mpCNMTMGlob"]

    for Imp in ImpLst:
        pCur = Imp[0]

        NumRho = Imp[1]
        delRho = 1/NumRho
        rho = delRho
        NumSampsESS = Imp[2]
        NumChainsESS = Imp[3]
        MVarComp = Imp[4]
    
        print("Currently running: p=" + str(pCur))
        print("Delta rho: " + str(delRho))
        print("Number of Samples per Chain to compute ESS/MSJD: " + str(NumSampsESS))
        print("Number of Independent Chain to compute ESS/MSJD: " + str(NumChainsESS))
        print("N in bar{g}_N^rho = 1/N sum_{j =1}^N g(X_jrho): " + str(NSampsingN))
        print("Number of separate chain M to compute Var(bar{g}_Nrho): " + str(MVarComp))

        rhoLstCur = []

        rhoCur= rho
        for curRnInx in range(0,NumRho):
            if rhoCur < .999:
                rhoLstCur.append(rhoCur)
            rhoCur += delRho

        prhoList.append([pCur, rhoLstCur])


        q0Lst = q0FN(NumChainsESS+MVarComp)
        
    

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:

            #Compute statistics for rho = 0
            #rhoz = 0.0
            #CurfNinput = (q0Lst,TargetDim,CovPrior,Pot,rhoz,pCur,NumSampsESS,NumChainsESS,NSampsingN,MVarComp,MomLen,SampleLnSv)

            #sample_run_dict[pool.submit(MCMCsmp.mixMetricsmPCN,*CurfNinput)] = (pCur,rhoz,algLst[0])
            #sample_run_dict[pool.submit(MCMCsmp.mixMetricsBBMTM,*CurfNinput)] = (pCur,rhoz,algLst[2])
            
            for rhoCur in rhoLstCur:

                CurfNinput = (q0Lst,TargetDim,CovPrior,Pot,rhoCur,pCur,NumSampsESS,NumChainsESS,NSampsingN,MVarComp,MomLen,SampleLnSv)
                
                sample_run_dict[pool.submit(mixMetricsmPCN,*CurfNinput)] =(pCur,rhoCur,algLst[0])
                sample_run_dict[pool.submit(mixMetricsmlocPCNMTM,*CurfNinput)]= (pCur,rhoCur,algLst[1])
                sample_run_dict[pool.submit(mixMetricsBBMTM,*CurfNinput)] = (pCur,rhoCur,algLst[2])

            print("Total MCMC Runs Submitted: " + str(len(sample_run_dict)))

            for f in tqdm(as_completed(sample_run_dict), total=len(sample_run_dict), desc="Parallel MCMC Runs"):
                prho_study_data_dict[sample_run_dict[f]] =  f.result()

        prho_study_data_dict[pCur,"ESS Chain Length"] = NumSampsESS
        prho_study_data_dict[pCur,"ESS Indep Chains"] = NumChainsESS

    prho_study_data_dict["Moment Estimator Length"] = MomLen
    prho_study_data_dict["p rho values List"] = prhoList #save list of p values with associated rho sweaps
    if saveDictExt:
        np.save(saveDictLoc, prho_study_data_dict, allow_pickle=True)  #Saving dictionary
    return prho_study_data_dict

def parameter_sweep_p_rho_save_figures(prho_study_data_dict,TarDim , FileNmBase):
    fig_colors = generate_colors(TarDim)
    algLst = ["mpCNOG","mpCNMTMLoc", "mpCNMTMGlob"]
 
    
    MomLen = prho_study_data_dict["Moment Estimator Length"]
    
    prhoList = prho_study_data_dict["p rho values List"]

    #Generate figures for ESS/N vs rho for each component of the posterior
    for curprhoLst in tqdm(prhoList, desc= "Building ESS and MSJD Graphics"):
        pcur = curprhoLst[0]
        rhoLst = curprhoLst[1]

        #ESS Lists
        ESSLstOG = []
        ESSLstLoc = []
        ESSLstGlob = []

        #MSJD Lists
        MSJDLstOG = []
        MSJDLstLoc = []
        MSJDLstGlob = []

        #Sample Means
        SMgNLstOG = []
        SMgNLstLoc = []
        SMgNLstGlob = []

        #Sample Covariance
        CovgNLstOG = []
        CovgNLstLoc = []
        CovgNLstGlob = []    

        #Sample Moments
        MomgNLstOG = []
        MomgNLstLoc = []
        MomgNLstGlob = []   

  
    
        delrho = rhoLst[1]- rhoLst[0]
        NumSampsESS = prho_study_data_dict[pcur,"ESS Chain Length"]
        NumChains = prho_study_data_dict[pcur,"ESS Indep Chains"] 
        curRunData ="p_" + str(pcur) + "_drho_" + str(delrho) + "_NSamps_" + str(NumSampsESS) + "_NChains_" + str(NumChains)
        for rho in rhoLst:
            #Retrieve ESS for different methods
            ESSLstOG.append(prho_study_data_dict[pcur,rho,algLst[0]]["ESS"])
            ESSLstLoc.append(prho_study_data_dict[pcur,rho,algLst[1]]["ESS"])
            ESSLstGlob.append(prho_study_data_dict[pcur,rho,algLst[2]]["ESS"])

            #Retrieve MSJD for different methods
            MSJDLstOG.append(prho_study_data_dict[pcur,rho,algLst[0]]["MSJD"])
            MSJDLstLoc.append(prho_study_data_dict[pcur,rho,algLst[1]]["MSJD"])
            MSJDLstGlob.append(prho_study_data_dict[pcur,rho,algLst[2]]["MSJD"])

            #Retrieve g_N^rho for different method
            #Sample Means
            SMgNLstOG.append(prho_study_data_dict[pcur,rho,algLst[0]]["Var(gNmean)"])
            SMgNLstLoc.append(prho_study_data_dict[pcur,rho,algLst[1]]["Var(gNmean)"])
            SMgNLstGlob.append(prho_study_data_dict[pcur,rho,algLst[2]]["Var(gNmean)"])

            #Sample Covariance
            CovgNLstOG.append(prho_study_data_dict[pcur,rho,algLst[0]]["Var(gNCov)"])
            CovgNLstLoc.append(prho_study_data_dict[pcur,rho,algLst[1]]["Var(gNCov)"])
            CovgNLstGlob.append(prho_study_data_dict[pcur,rho,algLst[2]]["Var(gNCov)"])

            #Sample Moments
            MomgNLstOG.append(prho_study_data_dict[pcur,rho,algLst[0]]["Var(gNMoms)"])
            MomgNLstLoc.append(prho_study_data_dict[pcur,rho,algLst[1]]["Var(gNMoms)"])
            MomgNLstGlob.append(prho_study_data_dict[pcur,rho,algLst[2]]["Var(gNMoms)"])

            #MomLen = prho_study_data_dict[pcur,rho,algLst[2]]["num Moms Saved"]


            #Retrieve and Plot time series data
            curTSOGData = prho_study_data_dict[pcur,rho,algLst[0]]["time_series"]
            curTSLocData = prho_study_data_dict[pcur,rho,algLst[1]]["time_series"]
            curTSGlobData = prho_study_data_dict[pcur,rho,algLst[2]]["time_series"]
            curRunDataTS= "time_series_p_" + str(pcur) + "/rho_" + str(round(rho,3))
            plot_timeseries(curTSOGData, FileNmBase + curRunDataTS+ "_mpCN.png", "mpCN" ,burn_in=1000)
            plot_timeseries(curTSLocData, FileNmBase + curRunDataTS+ "_mpCN_Loc.png", "mpCN MTM Local" ,burn_in=1000)
            plot_timeseries(curTSGlobData, FileNmBase + curRunDataTS+ "_mpCN_Glob.png", "mpCN MTM Global" ,burn_in=1000)


    
        for parmIndx in range(0,TarDim):
            #Plot ESS vs rho current p for different components
            #print(ESSLstOG)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$ESS/N$")
            ax.set_title(r"Mixing as measured by ESS/N for p ="+str(pcur)+" for Parameter Index " + str(parmIndx))
            ax.grid(alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(FileNmBase + curRunData+ "_ESS_vs_rho_ParaIndx_" + str(parmIndx)+ ".png")
            plt.close(fig)

            #print(SMgNLstOG)
            #Plot Var(g_N\rho) vs rho for current p for different components
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rhoLst, [gNcur[parmIndx] for gNcur in SMgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
            ax.plot(rhoLst, [gNcur[parmIndx] for gNcur in SMgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
            ax.plot(rhoLst, [gNcur[parmIndx] for gNcur in SMgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$v_j := \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N x_j^{(k)})$")
            ax.set_title(r"Mixing as measured by v_j for p = "+str(pcur)+" for j = " + str(parmIndx))
            ax.grid(alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(FileNmBase + curRunData+ "var_mean_Var_gNrho_x_j_vs_rho_ParaIndx_" + str(parmIndx)+ ".png")
            plt.close(fig)

        
        #Generate a plot for the min componental ESS 
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhoLst, [esslst.min()/NumSampsESS for esslst in ESSLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
        ax.plot(rhoLst, [esslst.min()/NumSampsESS for esslst in ESSLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
        ax.plot(rhoLst, [esslst.min()/NumSampsESS for esslst in ESSLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\min_j ESS_j/N$")
        ax.set_title(r"Mixing as measured by $\min_j ESS_j/N$ p = " +str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "_ESS_comp_min_vs_rho.png")
        plt.close(fig)

        #print(SMgNLstOG)
        #Plot max(Var(gN(means)\rho) vs rho for current p for different components
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhoLst, [gNcur.max() for gNcur in SMgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
        ax.plot(rhoLst, [gNcur.max() for gNcur in SMgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
        ax.plot(rhoLst, [gNcur.max() for gNcur in SMgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$v := \max_j \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N x_j^{(k)})$")
        ax.set_title(r"Mixing as measured by v for p = "+str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "var_mean_max_j_Var_gNrho_x_j_vs_rho" + str(parmIndx)+ ".png")
        plt.close(fig)


        #Plot max(Var(gN(cov)\rho) vs rho for current p
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhoLst, [gNcur.max() for gNcur in CovgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
        ax.plot(rhoLst, [gNcur.max() for gNcur in CovgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
        ax.plot(rhoLst, [gNcur.max() for gNcur in CovgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$v := \max_{i,j} \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N x_j^{(k)} x_i^{(k)})$")
        ax.set_title(r"Mixing as measured by v for p = "+str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "var_cov_max_i_j_Var_gNrho_x_j_x_i_vs_rho" + str(parmIndx)+ ".png")
        plt.close(fig)

        #Plot max(Var(gN(cov)\rho) vs rho for current p
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhoLst, [gNcur[0][1] for gNcur in CovgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
        ax.plot(rhoLst, [gNcur[0][1] for gNcur in CovgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
        ax.plot(rhoLst, [gNcur[0][1] for gNcur in CovgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$v := \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N x_1^{(k)} x_2^{(k)})$")
        ax.set_title(r"Mixing as measured by v for p = "+str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "var_cov_Var_gNrho_x_1_x_2_vs_rho" + str(parmIndx)+ ".png")
        plt.close(fig)

        #Plot max(Var(gN(mom)\rho) vs rho for current p
        for curMom in range(0,MomLen):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rhoLst, [gNcur[curMom] for gNcur in MomgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
            ax.plot(rhoLst, [gNcur[curMom] for gNcur in MomgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
            ax.plot(rhoLst, [gNcur[curMom] for gNcur in MomgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$v_m := \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N |x^{(k)}|^m)$")
            ax.set_title(r"Mixing as measured by v_m for p = "+str(pcur)+r"m = "+str(curMom+1))
            ax.grid(alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(FileNmBase + curRunData+ "var_mom_Var_gNrho_xm_vs_rho_m_" + str(curMom+1)+ ".png")
            plt.close(fig)
    
    
        #Generate figures for ESS/N comparing different components for mpCN
        fig, ax = plt.subplots(figsize=(6, 4))
        for parmIndx in range(0,TarDim):
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstOG], linestyle="-", label="cmp_"+str(parmIndx), color=fig_colors[parmIndx])
    
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$ESS/N$")
        ax.set_title(r"Mixing for mpCN as measured by ESS/N for p ="+str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "_ESS_v_rho_mpCN.png")
        plt.close(fig)

    
        #Generate figures for ESS/N comparing different components for mpCNMTMloc
        fig, ax = plt.subplots(figsize=(6, 4))
        for parmIndx in range(0,TarDim):
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstLoc], linestyle="-", label="cmp_"+str(parmIndx), color=fig_colors[parmIndx])
    
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$ESS/N$")
        ax.set_title(r"Mixing for mpCN_loc_MTM as measured by ESS/N for p ="+str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "_ESS_v_rho_mpCN_MTM_loc.png")
        plt.close(fig)

        #Generate figures for ESS/N comparing different components for mpCNMTM
        fig, ax = plt.subplots(figsize=(6, 4))
        for parmIndx in range(0,TarDim):
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstGlob], linestyle="-", label="cmp_"+str(parmIndx), color=fig_colors[parmIndx])
    
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$ESS/N$")
        ax.set_title(r"Mixing for mpCN_MTM as measured by ESS/N for p ="+str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "_ESS_v_rho_mpCN_MTM.png")
        plt.close(fig)


        #Finally generate figures for MSJD for different methods
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(rhoLst, MSJDLstOG, linestyle="-",label=r"$mpCN$", color="tab:orange")
        ax.plot(rhoLst, MSJDLstLoc, linestyle="-",label=r"$mpCNlocMTM$", color="tab:green")
        ax.plot(rhoLst, MSJDLstGlob, linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

        ax.set_xlabel(r"rho")
        ax.set_ylabel(r"MSJD")
        ax.set_title(r"Mixing as measured by MSJD for p ="+str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FileNmBase + curRunData+ "_MSJD_v_rho.png")
        plt.close(fig)


## Numerical Set-up for Problem Type A

def PotGaussPert(x, TarDim, PertDim, PriorCovInv, PostMean, PostCovInv, mode = None):
    xtrun = x[0:PertDim]
    return 1/2* xtrun.T @( PostCovInv - PriorCovInv) @ xtrun - xtrun.T@ PostCovInv @PostMean

#Potential Functions for Model Problem B

#Example Problem 1
#f(x,y) = (x-a)^p y 

def PotEx1(X,sig, a,r,z, mode = None):
    return (2* sig**2)**(-1) *(X[1]*(X[0] - a)**(r) - z)**2

#Example Problem 2
#f(x,y) = (x - x0)^2/a^2 +(y - y0)^2/b^2

def PotEx2(X, x0, y0, asqI, bsqI, sig, zdata, mode = None): 
    frwdX = asqI*(X[0] - x0)**2  + bsqI*(X[1]-y0)**2
    return (2* sig**2)**(-1) *(frwdX - zdata )**2    


#Numberical Set-up for Model Problem C



def MkAD_A_Mat(ModDim, curApar):
    """
    Generates an antisymmetric matric with the given Model Parameters
    """
    A = np.zeros([ModDim,ModDim],dtype=float)
    iju = np.triu_indices(ModDim, k=1)     # (i,j) for i<j
    #triu_indices returns the indices of all the above diagonal indicies
    A[iju] = curApar 
    A[(iju[1], iju[0])] = -curApar
    return A 
    ###NEW see old version if needed


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
    r"""
    A measure of the stiffness of a problem of the form
        mu(dq) = exp( - Pot(q))mu_0(dq) where mu_0 = N(0, Cov)
    Returns (an approximation of)
        M = \int Pot(x) \mu_0(dx),  \int (Pot(x) -M)^2 \mu_0(dx)
    """
    priorSmpLst = np.random.multivariate_normal(np.zeros(dim), Cov, size=L)
    PotValLst = [Pot(priorSmp) for priorSmp in priorSmpLst]
    return np.mean(PotValLst), np.var(PotValLst, ddof=0)




        


