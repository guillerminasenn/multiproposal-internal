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
        # Center proposal then draw a cloud of NProps candidates around it.
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

def MpCN_DATA(q0,dim,Cov,rho,Pot,NProps,L):
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
    nmRjt = 0.0
    rng = np.random.default_rng()
    samp = np.empty((L + 1, dim), dtype=float) #Make an array for the samples x^{(k)}
    sampPhi = np.empty((L+ 1, 1), dtype=float) #Make an array for the phi of the samples \phi(x^{(k)}
    samp[0] = np.array(q0)
    Cov_chol = np.linalg.cholesky(Cov) #Find E such that EE^* = C
    eta = np.sqrt(1.0 - rho * rho)
    for samID in range(1, L + 1):
        # Same sampler as MpCN, but store potentials and acceptance statistics.
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
        sampPhi[samID] = logAcp[idx].copy()
        nmRjt += int(idx == 0)

    my_dict_mpCN ={}
    my_dict_mpCN["samples"] = samp
    my_dict_mpCN["Pot(samples)"] = sampPhi 
    my_dict_mpCN["AR"] = nmRjt/L 
    return my_dict_mpCN



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

def locMpCNMTM_DATA(q0,dim,Cov,rho,Pot,NProps,L):
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
    sampPhi = np.empty((L+ 1, 1), dtype=float) #Make an array for the phi of the samples \phi(x^{(k)}
    samp[0] = np.array(q0)
    Cov_chol = np.linalg.cholesky(Cov) 
    #Find E such that EE^* = C
    eta = np.sqrt(1.0 - rho * rho)
    # Scaling prefactor arises from pCN proposal symmetry correction.
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


        # MTM backward step: draw reverse proposals around the candidate.
        
        mtmPropsAux = (rho* mtmProp[:,None] + eta * Cov_chol @ rng.standard_normal((dim,NProps -1))).T 
        #MTM Reverse Proproposal
        logAcpDenomMTM = np.empty(NProps, dtype=float)
        logAcpDenomMTM[0] = -1*potpreFac*Pot(rhoinv*prevSam)
        for j in range(NProps -1):
            logAcpDenomMTM[j+1] = -1* potpreFac*Pot(rhoinv*mtmPropsAux[j])
        logAcpDen_max = np.max(logAcpDenomMTM)

        # Log acceptance ratio for MTM correction.
        propmtmPot = Pot(mtmProp)
        prevmtmPot = Pot(prevSam) 
        logMTMacpNum = -1*propmtmPot -potpreFac* Pot(rhoinv*prevSam) + np.log(Acp_norm) + logAcp_max
        logMTMacpDen = -1*prevmtmPot - potpreFac* Pot(rhoinv*mtmProp) + np.log(np.exp(logAcpDenomMTM - logAcpDen_max).sum()) + logAcpDen_max

        logMTMacp = min(0, logMTMacpNum -logMTMacpDen)
        MTMacp = np.exp(logMTMacp)
        
        if rng.random() < MTMacp:
            samp[samID] = mtmProp
            sampPhi[samID] = propmtmPot
        else:
            nmRjt += 1
            samp[samID] = prevSam
            sampPhi[samID] = prevmtmPot

        
    my_dict_mpCNMTMLoc ={}
    my_dict_mpCNMTMLoc["samples"] = samp
    my_dict_mpCNMTMLoc["Pot(samples)"] = sampPhi 
    my_dict_mpCNMTMLoc["AR"] = nmRjt/L 
    return my_dict_mpCNMTMLoc


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



def MpCNBBMTM_DATA(q0,dim,Cov,rho,Pot,NProps,L):
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
    sampPhi = np.empty((L+ 1, 1), dtype=float) #Make an array for the phi of the samples \phi(x^{(k)}
    samp[0] = np.array(q0)
    Cov_chol = np.linalg.cholesky(Cov) 
    #Find E such that EE^* = C
    eta = np.sqrt(1.0 - rho * rho)
    for samID in range(1, L + 1):
        # Bubble-bath MTM: proposals centered at current state.
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

        #Log Acceptance
        logMTMacpNum = np.log(Acp_norm) +logAcp_max
        logMTMacpDen = np.log(np.exp(logAcpDenomMTM - logAcpDen_max).sum()) + logAcpDen_max
        
        logMTMacp = min(0, logMTMacpNum -logMTMacpDen)
        MTMacp = np.exp(logMTMacp)
        
        if rng.random() < MTMacp:
            samp[samID] = mtmProp
            sampPhi[samID] = -1*logAcpDenomMTM[idx]
        else:
            nmRjt += 1
            samp[samID] = prevSam
            sampPhi[samID] = -1*logAcpDenomMTM[0]
        
    my_dict_mpCNMTM ={}
    my_dict_mpCNMTM["samples"] = samp
    my_dict_mpCNMTM["Pot(samples)"] = sampPhi 
    my_dict_mpCNMTM["AR"] = nmRjt/L 
    return my_dict_mpCNMTM






        


