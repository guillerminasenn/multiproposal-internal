#ImportMCMCSampliers
import MCMC_Sampliers_Testing as MCMCsmp

# #To run on multiple cores
# from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import Event

#Numerical Elements
# from math import exp 
from numpy.linalg import norm
import numpy as np
# from numpy import dot, array, transpose, diag
# import random
# import math

#Input Output utils
import os
import pandas as pd

#Stats elements
from scipy.stats import norm as normdist
# from arviz import ess
import arviz as az
import xarray as xr

#Plotting stuff
from matplotlib.patches import Ellipse

#Sending Messages
import smtplib
from email.message import EmailMessage
import requests

#Keeping track of time
import time
from datetime import timedelta

import warnings
warnings.simplefilter("error", RuntimeWarning)





# Send finished message (SMS via email gateway)
def message_text(subject,message):
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




def message_ntfy(msg):
    requests.post(
        "https://ntfy.sh/negh_MCMC_runs_04_23_86_07_14_20_9_10_23",
        data=msg.encode("utf-8"),
        timeout=5
    )



def analyze_MCMC_method(MCMC_meth, q0_lst, scm_args, NumParms, ESS_MSJD_Chain_len, ESS_MSJD_runs, Nave, NaveRuns, numMom, tsLen):
    r"""
    -MCMC_meth: MCMC Method that take an initial point as its first arg, a length of chain L 
    and will spit out a dictionary of the form
        my_dict_mcmc["samples"] = samp
        my_dict_mcmc["Pot(samples)"] = sampPhi 
        my_dict_mcmc["AR"] = nmRjt/L 
    -q0_lst: is intended as an i.i.d list of initial condition from pi of length numRuns + ESSruns
    -scm_args: intended as a list of argument to the MCMC method
    -ESS_MSJD_Chain_len: Length of chain to compute ESS/MSJD 
    -ESS_MSJD_runs: Number of independent ESS/MSJD chains
    -Nave: size of runs N for bar{g}_N^\rho = 1/N \sum_{k =1}^N
    -NaveRuns: number of independent run-- intended start from mu
    -numMom: Number of moments to compute for the observables
    -saveLn: How long to save the chain to give chain time series diagonistics
    """
    
    # Estimate MSJD and ESS across multiple independent chains.
    MSJDdiff_sum = 0.0
    ESS_Runs = []
    for parmIndx in range(0,ESS_MSJD_runs):
        MCMCargs = [q0_lst[parmIndx]]+ scm_args +[ESS_MSJD_Chain_len]
        run_dict = MCMC_meth(*MCMCargs)
        curSamp = run_dict["samples"]
        curSampPot = run_dict["Pot(samples)"]
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
    MSJD = MSJDdiff_sum/((ESS_MSJD_Chain_len-1)*ESS_MSJD_runs)

    # Monte Carlo estimators for means, covariances, moments, and potential.
    gNmeans = np.empty((NaveRuns, NumParms), dtype=float)          # each run's mean vector
    gNCov   = np.empty((NaveRuns, NumParms, NumParms), dtype=float)       # each run's sample cov matrix
    gNMom   = np.empty((NaveRuns, numMom), dtype=float)          # each run's moments (norm^j)
    gNPhi   = np.empty(NaveRuns,dtype=float)

    for parmIndx in range(0,NaveRuns):
        MCMCargs = [q0_lst[parmIndx]]+ scm_args +[Nave]
        run_dictgN = MCMC_meth(*MCMCargs)
        curSampgN = run_dictgN["samples"]
        gNmeans[parmIndx] = curSampgN.mean(axis=0)
        gNCov[parmIndx] = np.cov(curSampgN, rowvar=False, bias=False)
        gNnorms = np.linalg.norm(curSampgN, axis=1)
        for j in range(1, numMom + 1):
            gNMom[parmIndx, j-1] = np.mean(gNnorms**j)
        gNPhi[parmIndx] = run_dictgN["Pot(samples)"].mean()
        
    var_gNmean = gNmeans.var(axis=0, ddof=1)
    var_gNCov_entries = gNCov.var(axis=0, ddof=1) 
    var_gNMom = gNMom.var(axis=0, ddof=1)
    var_gNPhi = gNPhi.var(ddof=1)

    means_gNmean = gNmeans.mean(axis=0)
    means_gNCov_entries = gNCov.mean(axis=0) 
    means_gNMom = gNMom.mean(axis=0)
    means_gNPhi = gNPhi.mean()

    #var_gNWTN = gNWTN.var(ddof=1) 


    # return a dictionary of with run parameters + 
    # ESS, MSJD, g_N = 1/N \sum_{j =1}^N g(X_j) for various g, and some samples 
    my_big_dict = {}
    my_big_dict["ESS_Smp_Size"] = ESS_MSJD_Chain_len
    my_big_dict["ESS_Ind_Runs"] = ESS_MSJD_runs
    my_big_dict["Estimator_Ln"] = Nave
    my_big_dict["num_Estimator_runs"] = NaveRuns
    my_big_dict["ESS"] = ess_vec
    my_big_dict["MSJD"] = MSJD
    my_big_dict["Var(gNmean)"] = var_gNmean
    my_big_dict["Var(gNCov)"] = var_gNCov_entries
    my_big_dict["Var(gNMoms)"] = var_gNMom
    my_big_dict["Var(gNPot)"] = var_gNPhi
    my_big_dict["mean(gNmean)"] = means_gNmean
    my_big_dict["mean(gNCov)"] = means_gNCov_entries
    my_big_dict["mean(gNMoms)"] = means_gNMom
    my_big_dict["mean(gNPot)"] = means_gNPhi
    my_big_dict["num Moms Saved"] = numMom
    #my_big_dict["var(gNWtnMean)"] = var_gNWTN
    my_big_dict["time_series"] = curSamp[0:tsLen]
    my_big_dict["time_series_Pot"] = curSampPot[0:tsLen]
    return my_big_dict 


def parameter_sweep_p_rho(ImpLst, q0FN, TargetDim, CovPrior,Pot, SampleLnSv = 2000, NSampsingN = 100, MomLen = 4, messageUpdates = True, saveDictExt = False, saveDictLoc = False):
    r"""
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

        q0FN --Function to Randomly sample initial list from target mu
    """
    
    sample_run_dict ={} #Make an empty dictionary to store sample runs
    prho_study_data_dict ={} #Make an empty dictionary to store all outputs of p/rho sweep

    prho_study_data_dict["Input List"] = ImpLst #save input list
    prhoList = []
    algLst = ["mpCNOG","mpCNMTMLoc", "mpCNMTMGlob"]

    for Imp in ImpLst:
        time_start = time.perf_counter()
        
        pCur = Imp[0]

        NumRho = Imp[1]
        delRho = 1/NumRho
        rho = delRho
        NumSampsESS = Imp[2]
        NumChainsESS = Imp[3]
        MVarComp = Imp[4]

        currInfoStr = "Currently running: p=" + str(pCur) + "\n"
        currInfoStr += "Delta rho: " + str(delRho) + "\n"
        currInfoStr += "Number of Samples per Chain to compute ESS/MSJD: " + str(NumSampsESS) + "\n"
        currInfoStr += "Number of Independent Chain to compute ESS/MSJD: " + str(NumChainsESS) + "\n"
        currInfoStr += "N in bar{g}_N^rho = 1/N sum_{j =1}^N g(X_jrho): " + str(NSampsingN) + "\n"
        currInfoStr += "Number of separate chain M to compute Var(bar{g}_Nrho): " + str(MVarComp)
        print(currInfoStr)
        if messageUpdates:
            message_ntfy(currInfoStr)      
        
        #print("Currently running: p=" + str(pCur))
        #print("Delta rho: " + str(delRho))
        #print("Number of Samples per Chain to compute ESS/MSJD: " + str(NumSampsESS))
        #print("Number of Independent Chain to compute ESS/MSJD: " + str(NumChainsESS))
        #print("N in bar{g}_N^rho = 1/N sum_{j =1}^N g(X_jrho): " + str(NSampsingN))
        #print("Number of separate chain M to compute Var(bar{g}_Nrho): " + str(MVarComp))

        rhoLstCur = []

        rhoCur= rho
        for curRnInx in range(0,NumRho):
            if rhoCur < .999:
                rhoLstCur.append(rhoCur)
            rhoCur += delRho

        prhoList.append([pCur, rhoLstCur])
        
    

        # Run mpCN variants in parallel for each rho value.
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:

            for rhoCur in rhoLstCur:

                q0Lst = q0FN(NumChainsESS+MVarComp)
                mcmc_Args = [TargetDim,CovPrior,rhoCur,Pot,pCur]
                CurfNinput = [q0Lst, mcmc_Args,TargetDim,NumSampsESS,NumChainsESS,NSampsingN,MVarComp,MomLen,SampleLnSv]

                mpCNarg= [MCMCsmp.MpCN_DATA]+CurfNinput
                sample_run_dict[pool.submit(analyze_MCMC_method,*mpCNarg)] =(pCur,rhoCur,algLst[0])
                
                mpCNMTMLocarg= [MCMCsmp.locMpCNMTM_DATA]+CurfNinput
                sample_run_dict[pool.submit(analyze_MCMC_method,*mpCNMTMLocarg)] = (pCur,rhoCur,algLst[1])

                mpCNMTMGlobarg= [MCMCsmp.MpCNBBMTM_DATA]+CurfNinput               
                sample_run_dict[pool.submit(analyze_MCMC_method,*mpCNMTMGlobarg)] = (pCur,rhoCur,algLst[2])

            print("Total MCMC Runs Submitted: " + str(len(sample_run_dict)))
            if messageUpdates:
                message_ntfy("Total MCMC Runs Submitted: " + str(len(sample_run_dict))) 

            for f in tqdm(as_completed(sample_run_dict), total=len(sample_run_dict), desc="Parallel MCMC Runs"):
                prho_study_data_dict[sample_run_dict[f]] =  f.result()

        prho_study_data_dict[pCur,"ESS Chain Length"] = NumSampsESS
        prho_study_data_dict[pCur,"ESS Indep Chains"] = NumChainsESS
        prho_study_data_dict[pCur,"gN Average Length"] = NSampsingN
        prho_study_data_dict[pCur,"gN Indep Chains"] = MVarComp

        time_end = time.perf_counter()
        run_time = time_end - time_start
        print("Total Run Time Was: " + str(timedelta(seconds=run_time)))
        if messageUpdates:
            message_ntfy("p = " + str(pCur) + " is complete.")
            message_ntfy("Total Run Time Was: " + str(timedelta(seconds=run_time)))
                         
    prho_study_data_dict["Moment Estimator Length"] = MomLen
    prho_study_data_dict["p rho values List"] = prhoList #save list of p values with associated rho sweaps
    if saveDictExt:
        np.save(saveDictLoc, prho_study_data_dict, allow_pickle=True)  #Saving dictionary
    return prho_study_data_dict


def writeCSV(filenm, sampArray, newFile = False):
    if newFile:
        pd.DataFrame(sampArray).to_csv(filenm, mode='w', index=False, header=False)
    else:
        pd.DataFrame(sampArray).to_csv(filenm, mode='a', index=False, header=False)

def readCSV(filenm):   
    df = pd.read_csv(filenm)
    MpCNsamp2 = df.to_numpy()
    return MpCNsamp2

# Fun Progress Bar (auto-falls back to terminal when widgets are unavailable)
from tqdm.auto import tqdm

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
                    z = normdist.ppf((1 + beta) / 2)
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


def makeHistGrid_Comps(R, dr, sampList, compLst,saveLoc, C=None, beta=0.95, hidePlt = True):
    # Extract the requested components and plot marginal and pairwise histograms.
    SampLstsGd = [getComp(sampList,j) for j in compLst]
    numbins= int(2*R/dr)
    x_bins = np.linspace(-R, R, numbins) 
    y_bins = np.linspace(-R, R, numbins)

    NumParams = len(compLst)
    fig, axs = plt.subplots(NumParams, NumParams,figsize=(15,15))
    for i in range(0,NumParams):
        for j in range(0,NumParams):
            if i == j:
                axs[i,j].hist(SampLstsGd[i], density=True, bins=x_bins)
                # Add confidence interval ticks if covariance provided
                if C is not None:
                    # Quantile of standard normal
                    z = normdist.ppf((1 + beta) / 2)
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




def plot_ESS(prho_study_data_dict,TarDim, fileLoc):
    fig_colors = generate_colors(TarDim)
    algLst = ["mpCNOG","mpCNMTMLoc", "mpCNMTMGlob"]
    prhoList = prho_study_data_dict["p rho values List"]
    
    for curprhoLst in tqdm(prhoList, desc= "Building ESS Plots"):
        pcur = curprhoLst[0]
        rhoLst = curprhoLst[1]

        #ESS Lists
        ESSLstOG = []
        ESSLstLoc = []
        ESSLstGlob = []
        for rho in rhoLst:
            #Retrieve ESS for different methods
            ESSLstOG.append(prho_study_data_dict[pcur,rho,algLst[0]]["ESS"])
            ESSLstLoc.append(prho_study_data_dict[pcur,rho,algLst[1]]["ESS"])
            ESSLstGlob.append(prho_study_data_dict[pcur,rho,algLst[2]]["ESS"])
            
            delrho = rhoLst[1]- rhoLst[0]
            NumSampsESS = prho_study_data_dict[pcur,"ESS Chain Length"]
            NumChains = prho_study_data_dict[pcur,"ESS Indep Chains"] 
            curRunData ="p_" + str(pcur) + "_drho_" + str(delrho) + "_NSamps_" + str(NumSampsESS) + "_NChains_" + str(NumChains)


        for parmIndx in range(0,TarDim):
            #Plot ESS vs rho current p for different components
            #print(ESSLstOG)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
            ax.plot(rhoLst, [esslst[parmIndx]/NumSampsESS for esslst in ESSLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$ESS/N$")
            ax.set_title(r"Mixing as measured by ESS/N for p = " +str(pcur)+ " for Parameter Index " + str(parmIndx))
            ax.grid(alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(fileLoc + curRunData+ "_ESS_vs_rho_ParaIndx_" + str(parmIndx)+ ".png")
            plt.close(fig)

        #Generate a plot for the min componental ESS 
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhoLst, [esslst.min()/NumSampsESS for esslst in ESSLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
        ax.plot(rhoLst, [esslst.min()/NumSampsESS for esslst in ESSLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
        ax.plot(rhoLst, [esslst.min()/NumSampsESS for esslst in ESSLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\min_j \, ESS_j/N$")
        ax.set_title(r"Mixing as measured by $\min_j \, \frac{ESS_j}{N}$ p = " +str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(fileLoc + curRunData+ "_ESS_comp_min_vs_rho.png")
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
        plt.savefig(fileLoc + curRunData+ "_ESS_v_rho_mpCN.png")
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
        plt.savefig(fileLoc + curRunData+ "_ESS_v_rho_mpCN_MTM_loc.png")
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
        plt.savefig(fileLoc + curRunData+ "_ESS_v_rho_mpCN_MTM.png")
        plt.close(fig)


def plot_MSDJ(prho_study_data_dict, fileLoc):
    algLst = ["mpCNOG","mpCNMTMLoc", "mpCNMTMGlob"]
    prhoList = prho_study_data_dict["p rho values List"]
    for curprhoLst in tqdm(prhoList, desc= "Building MSJD Plots"):
        pcur = curprhoLst[0]
        rhoLst = curprhoLst[1]
        
        #MSJD Lists
        MSJDLstOG = []
        MSJDLstLoc = []
        MSJDLstGlob = []

        delrho = rhoLst[1]- rhoLst[0]
        NumSampsESS = prho_study_data_dict[pcur,"ESS Chain Length"]
        NumChains = prho_study_data_dict[pcur,"ESS Indep Chains"] 
        curRunData ="p_" + str(pcur) + "_drho_" + str(delrho) + "_NSamps_" + str(NumSampsESS) + "_NChains_" + str(NumChains)

        for rho in rhoLst:
            MSJDLstOG.append(prho_study_data_dict[pcur, rho, algLst[0]]["MSJD"])
            MSJDLstLoc.append(prho_study_data_dict[pcur, rho, algLst[1]]["MSJD"])
            MSJDLstGlob.append(prho_study_data_dict[pcur, rho, algLst[2]]["MSJD"])

        #Finally generate figures for MSJD for different methods
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhoLst, MSJDLstOG,  linestyle="-", label=r"$mpCN$",        color="tab:orange")
        ax.plot(rhoLst, MSJDLstLoc, linestyle="-", label=r"$mpCNlocMTM$",  color="tab:green")
        ax.plot(rhoLst, MSJDLstGlob,linestyle="-", label=r"$mpCNMTM$",     color="tab:blue")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("MSJD")
        ax.set_title(f"Mixing (MSJD) vs $\\rho$ for p={pcur}")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(fileLoc + curRunData + "_MSJD_v_rho.png")
        plt.close(fig)
        

def plot_samp_vargN(prho_study_data_dict,TarDim, fileLoc):
    fig_colors = generate_colors(TarDim)
    algLst = ["mpCNOG","mpCNMTMLoc", "mpCNMTMGlob"]
    
    MomLen = prho_study_data_dict["Moment Estimator Length"]
    prhoList = prho_study_data_dict["p rho values List"]

    #Generate figures for ESS/N vs rho for each component of the posterior
    for curprhoLst in tqdm(prhoList, desc= "Building Plots Var(g_N) for different g_N"):
        pcur = curprhoLst[0]
        rhoLst = curprhoLst[1]
        
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


        #Sample Moments
        PotgNLstOG = []
        PotgNLstLoc = []
        PotgNLstGlob = [] 

        delrho = rhoLst[1]- rhoLst[0]

        N_len_gN = prho_study_data_dict[pcur,"gN Average Length"]
        M_indep_gN = prho_study_data_dict[pcur,"gN Indep Chains"]
        
        curRunData ="p_" + str(pcur) + "_drho_" + str(delrho) + "_gNLen_" + str(N_len_gN) + "_MReps_" + str(M_indep_gN)
        
        for rho in rhoLst:
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

            #Sample Phi
            PotgNLstOG.append(prho_study_data_dict[pcur,rho,algLst[0]]["Var(gNPot)"])
            PotgNLstLoc.append(prho_study_data_dict[pcur,rho,algLst[1]]["Var(gNPot)"])
            PotgNLstGlob.append(prho_study_data_dict[pcur,rho,algLst[2]]["Var(gNPot)"])
        
        for parmIndx_i in range(0,TarDim):
            #print(SMgNLstOG)
            #Plot Var(g_N\rho) vs rho for current p for different components
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rhoLst, [gNcur[parmIndx_i] for gNcur in SMgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
            ax.plot(rhoLst, [gNcur[parmIndx_i] for gNcur in SMgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
            ax.plot(rhoLst, [gNcur[parmIndx_i] for gNcur in SMgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$v_i := \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N x_i^{(k)})$")
            ax.set_title(r"Mixing as measured by v_i for p = "+str(pcur)+" for i = " + str(parmIndx_i))
            ax.grid(alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(fileLoc + curRunData+ "Var_gNrho_x_i_vs_rho_ParaIndx_" + str(parmIndx_i)+ ".png")
            plt.close(fig)

            for parmIndx_j in range(0,TarDim):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(rhoLst, [gNcur[parmIndx_i,parmIndx_j] for gNcur in CovgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
                ax.plot(rhoLst, [gNcur[parmIndx_i,parmIndx_j] for gNcur in CovgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
                ax.plot(rhoLst, [gNcur[parmIndx_i,parmIndx_j] for gNcur in CovgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

                ax.set_xlabel(r"$\rho$")
                ax.set_ylabel(r"$v_{i,j} := \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N x_i^{(k)} x_j^{(k)})$")
                ax.set_title(r"Mixing as measured by v_{i,j} for p = "+str(pcur)+" for i,j = " + str(parmIndx_i) + "," + str(parmIndx_j))
                ax.grid(alpha=0.3)
                ax.legend()

                plt.tight_layout()
                plt.savefig(fileLoc + curRunData+ "Var_gNrho_x_i_x_j_vs_rho_ParaIndx_" + str(parmIndx_i)+"_" + str(parmIndx_j)+".png")
                plt.close(fig)

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
        plt.savefig(fileLoc + curRunData+ "max_Var_gNrho_x_j_vs_rho.png")
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
        plt.savefig(fileLoc + curRunData+ "max_Var_gNrho_x_j_x_i_vs_rho.png")
        plt.close(fig)

        #Plot max(Var(gN(mom)\rho) vs rho for current p
        for curMom in range(0,MomLen):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rhoLst, [gNcur[curMom] for gNcur in MomgNLstOG], linestyle="-", label=r"$mpCN$", color="tab:orange")
            ax.plot(rhoLst, [gNcur[curMom] for gNcur in MomgNLstLoc], linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
            ax.plot(rhoLst, [gNcur[curMom] for gNcur in MomgNLstGlob], linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$v_m := \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N |x^{(k)}|^m)$")
            ax.set_title(r"Mixing as measured by v_m for p = "+str(pcur)+ " m = " + str(curMom+1))
            ax.grid(alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(fileLoc + curRunData+ "Var_gNrho_mom_vs_rho_m_" + str(curMom+1)+ ".png")
            plt.close(fig)


        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhoLst, PotgNLstOG, linestyle="-", label=r"$mpCN$", color="tab:orange")
        ax.plot(rhoLst, PotgNLstLoc, linestyle="-", label=r"$mpCNlocMTM$", color="tab:green")
        ax.plot(rhoLst, PotgNLstGlob, linestyle="-",label=r"$mpCNMTM$", color="tab:blue")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$v_\Phi := \mathrm{Var}(\frac{1}{N} \sum_{k=1}^N \Phi(x^{(k)}))$")
        ax.set_title(r"Mixing as measured by $v_\Phi$ for p = "+ str(pcur))
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(fileLoc + curRunData+ "Var_gNrho_Phi_vs_rho.png")
        plt.close(fig)



def plot_timeseries(samples, potsample, components, filename, MCMC_type, burn_in=0):
    r"""
    Plot time series (trace plots) for each component of the MCMC chain
    and save to a file.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (L+1, dim) of samples (x_0,..., x_L) returned by an MCMC method. 
        Each row is an iteration, each column is a component.
    potsample : np.ndarray
        Array of shape (L+1, 1) returned by an  (Pot(x_0),..., Pot(x_L)) a preconditioned MCMC method. 
        Each row is an iteration, each column is a component.
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
    
    # One trace for potential + one trace per selected component.
    subplotdim = len(components) +1
    # Make one subplot per dimension
    fig, axes = plt.subplots(subplotdim, 1, figsize=(8, 2.0 * dim), sharex=True)

    axes[0].plot(iters, potsample[burn_in:])
    axes[0].set_ylabel(f"$\\phi(x)$")
    axes[0].grid(alpha=0.3)
    
    for d in range(len(components)):
        axes[d+1].plot(iters, samples[burn_in:, components[d]])
        axes[d+1].set_ylabel(f"$x_{components[d]+1}$")
        axes[d+1].grid(alpha=0.3)

    axes[-1].set_xlabel("Iteration")

    fig.suptitle(" Trace Plots for " + MCMC_type, y=0.99)
    fig.tight_layout()

    # Ensure directory exists (if any directory is specified)
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save and close
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_samp_timeseries(prho_study_data_dict,TarDim, components, burnIn, fileLoc):
    algLst = ["mpCNOG","mpCNMTMLoc", "mpCNMTMGlob"]
    prhoList = prho_study_data_dict["p rho values List"]
    for curprhoLst in tqdm(prhoList, desc= "Building Time Series Plots"):
        pcur = curprhoLst[0]
        rhoLst = curprhoLst[1]
        for rho in rhoLst:
            #Retrieve and Plot time series data
            curTSOGData = prho_study_data_dict[pcur,rho,algLst[0]]["time_series"]
            curTSLocData = prho_study_data_dict[pcur,rho,algLst[1]]["time_series"]
            curTSGlobData = prho_study_data_dict[pcur,rho,algLst[2]]["time_series"]
            curTSOGDataPot = prho_study_data_dict[pcur,rho,algLst[0]]["time_series_Pot"]
            curTSLocDataPot = prho_study_data_dict[pcur,rho,algLst[1]]["time_series_Pot"]
            curTSGlobDataPot = prho_study_data_dict[pcur,rho,algLst[2]]["time_series_Pot"]
            
            curRunDataTS= "time_series_p_" + str(pcur) + "/rho_" + str(round(rho,3))
            plot_timeseries(curTSOGData, curTSOGDataPot,components, fileLoc + curRunDataTS+ "_mpCN.pdf", "mpCN" ,burn_in=burnIn)
            plot_timeseries(curTSLocData, curTSLocDataPot,components, fileLoc + curRunDataTS+ "_mpCN_Loc.pdf", "mpCN MTM Local" ,burn_in=burnIn)
            plot_timeseries(curTSGlobData, curTSGlobDataPot,components, fileLoc + curRunDataTS+ "_mpCN_Glob.pdf", "mpCN MTM Global" ,burn_in=burnIn)


def parameter_sweep_p_rho_save_figures(prho_study_data_dict,TarDim, components, burnIn, FileNmBase):

    fileLocEss = FileNmBase + "ESS/"
    os.makedirs(fileLocEss, exist_ok=True)
    plot_ESS(prho_study_data_dict,TarDim, fileLocEss)

    fileLocMSJD = FileNmBase + "MSJD/"
    os.makedirs(fileLocMSJD, exist_ok=True)
    plot_MSDJ(prho_study_data_dict, fileLocMSJD) 

    fileLocGn = FileNmBase + "vargN/"
    os.makedirs(fileLocGn, exist_ok=True)
    plot_samp_vargN(prho_study_data_dict,TarDim, fileLocGn)

    fileLocTS = FileNmBase + "time_series/"
    os.makedirs(fileLocTS, exist_ok=True)
    plot_samp_timeseries(prho_study_data_dict,TarDim, components, burnIn, fileLocTS)


def parallel_MCMC_Runs_ESS(chainLn,numChain,MCMCmeth,MCMCmethArgs, q0gen, burn_In):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
        MCMCsampRunJobs = []
        for run in range(0,numChain):
            q0zCur = q0gen()
            CurfNinput = [q0zCur] + MCMCmethArgs + [chainLn]
            MCMCsampRunJobs.append(pool.submit(MCMCmeth,*CurfNinput))


        print("Total MCMC Runs: " + str(len(MCMCsampRunJobs)))

        finished_chains = []
        for f in tqdm(as_completed(MCMCsampRunJobs), total=len(MCMCsampRunJobs), desc="Parallel MCMC Runs"):
            finished_chains.append(f.result()[burn_In:,:])

    #Compute ESS
    chainStack = np.stack(finished_chains, axis=0)     # (chain, draw, d)
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
    ess_vec = ess_da["x"].values                           
        
    return np.concatenate(finished_chains), ess_vec


def parallel_MCMC_Runs(chainLn, numChain, MCMCmeth, MCMCmethArgs, q0gen, burn_In, thin, max_workers=None):
    """
    Run many chain in parallel
        chainLn = Length of each chain
        numChain = number of parallel chains
    """
    if max_workers is None:
        max_workers = min(numChain, mp.cpu_count())
    else:
        max_workers = min(max_workers, numChain)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        MCMCsampRunJobs = []
        for run in range(0,numChain):
            q0zCur = q0gen()
            CurfNinput = [q0zCur] + MCMCmethArgs + [chainLn]
            MCMCsampRunJobs.append(pool.submit(MCMCmeth,*CurfNinput))


        print("Total MCMC Runs: " + str(len(MCMCsampRunJobs)))

        finished_chains = []
        for f in tqdm(as_completed(MCMCsampRunJobs), total=len(MCMCsampRunJobs), desc="Parallel MCMC Runs"):
            curChain = f.result()[burn_In:,:]
            finished_chains.append(curChain[::thin])                    
        
    return np.concatenate(finished_chains, axis = 0)


def parallel_MCMC_Runs_Data(chainLn, numChain, MCMCmeth, MCMCmethArgs, q0gen, burn_In, thin, max_workers=None):
    """
    Run many chain in parallel using on of the methods producing a dictionary of results
        chainLn = Length of each chain
        numChain = number of parallel chains
    """
    # Fan out independent chains across CPU cores, then concatenate.
    if max_workers is None:
        max_workers = min(numChain, mp.cpu_count())
    else:
        max_workers = min(max_workers, numChain)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        MCMCsampRunJobs = []
        for run in range(0,numChain):
            q0zCur = q0gen()
            CurfNinput = [q0zCur] + MCMCmethArgs + [chainLn]
            MCMCsampRunJobs.append(pool.submit(MCMCmeth,*CurfNinput))


        print("Total MCMC Runs: " + str(len(MCMCsampRunJobs)))

        finished_chains = []
        finished_Pot_chains = []
        AR_lst = []
        for f in tqdm(as_completed(MCMCsampRunJobs), total=len(MCMCsampRunJobs), desc="Parallel MCMC Runs"):
            curRes = f.result()
            curChain = curRes["samples"][burn_In:,:]
            curChainPot = curRes["Pot(samples)"][burn_In:,:]
            AR_lst.append(curRes["AR"])
            finished_chains.append(curChain[::thin])  
            finished_Pot_chains.append(curChainPot[::thin]) 
            
        
    return np.concatenate(finished_chains, axis = 0), np.concatenate(finished_Pot_chains, axis = 0), sum(AR_lst)/len(AR_lst)


#Code to generate a random orthogonal matrix (uniform from O(n)) using the QR factorization of a Guassian matrix.
def rndm_orth_matrix(n):

    #Generate a random n x n matrix with i.i.d. normal entries
    A = np.random.randn(n, n)
    
    #Perform the QR factorization
    Q, R = np.linalg.qr(A)    
    
    return Q



def generate_Random_Rot_Hist(numParms, MCMCsamps,FileNmBase, R = 5, dr = .1, numRots=50):
    for Rot_indx in tqdm(range(0,numRots), desc= "Generating Histograms from Randomly Rotated MCMC Runs"):

        RanRot = rndm_orth_matrix(numParms)

        MCMCsampsCurRot = [RanRot @ A for A in MCMCsamps]

        #Generate Histogram
        #Dimensions For Histogram Plot

        os.makedirs(FileNmBase+"Rotations/", exist_ok=True)
        histFileNm = FileNmBase + "Rotations/" +"Baseline_Histogram_Rot_" + str(Rot_indx) + ".pdf"
        makeHistGrid(R, dr, MCMCsampsCurRot, numParms, histFileNm, hidePlt =True)




    

## Numerical Set-up for Problem Type A
## Pot(x) = 1/2 x^t ( C_post^{-1} - C_prior^{-1}) - x^t C_post^{-1} m

def PotGaussPertCov(x, Pres_Diff, mode = None):
    return 1/2* x.T @( Pres_Diff) @ x

def PotGaussPertMean(x, Post_Mean, PostCovInv, mode = None):
    return - x.T@ PostCovInv @Post_Mean

def PotGaussPertFull(x,Pres_Diff, Post_Mean, PostCovInv, mode = None):
    return 1/2* x.T @( Pres_Diff) @ x - x.T@ PostCovInv @Post_Mean

#Potential Functions for Model Problem B

#Example Problem B1
#f(x,y) = (x-a)^p y 

def PotExB1(X,sig, a,r,z, mode = None):
    return (2* sig**2)**(-1) *(X[1]*(X[0] - a)**(r) - z)**2


#Example Problem B2
#f(x) = x^* C^{-1} x
def PotMahalanobis(X, compDim, sig, CovInv, zdata, mode = None):
    frwdX = float(X[0:compDim] @ CovInv @ X[0:compDim]) 
    return (2* sig**2)**(-1) *(frwdX - zdata )**2 



##### Model Problem C 

#General Numberical Set-up 

#def MkAD_A_Mat(ModDim, curApar):
#    """
#    Generates an antisymmetric matric with the given Model Parameters
#    """
#    A = np.zeros([ModDim,ModDim])
#    A[np.triu_indices(ModDim, k=1)] = curApar 
#    #triu_indices returns the indices of all the above diagonal indicies
#    return A - A.T

def MkAD_A_Mat(ModDim, curApar):
    """
    Generates an antisymmetric matrix A from the parameter vector (upper triangle).
    """
    A = np.zeros([ModDim,ModDim],dtype=float)
    iju = np.triu_indices(ModDim, k=1)     # (i,j) for i<j
    #triu_indices returns the indices of all the above diagonal indicies
    A[iju] = curApar 
    A[(iju[1], iju[0])] = -curApar
    return A 

def Apar_from_A(A):
    """Extract upper-triangular (k=1) entries of skew matrix A into vector Apar."""
    d = A.shape[0]
    iju = np.triu_indices(d, k=1)
    return A[iju].copy()


def ij_to_k(i, j, d):
    """
    Map matrix index (i,j) to parameter-space index k
    corresponding to np.triu_indices(d, k=1).

    Returns:
        k : int   (0 <= k < d*(d-1)//2)

    Raises:
        ValueError if i == j
    """
    if i == j:
        raise ValueError("Diagonal entries A[i,i] are zero and not parameterized.")

    # reduce to upper triangle
    if i < j:
        iu, ju = i, j
    else:
        iu, ju = j, i

    # number of upper-triangular entries before row iu
    k = iu*(d - 1) - iu*(iu - 1)//2 + (ju - iu - 1)
    return k

def getThA(ModDim, Apar, g, kappa):
    """
    Solve $(A + \kappa I)\theta = g$ for the state given the skew parameters.
    """
    A_p_kI = MkAD_A_Mat(ModDim, Apar)+ kappa*np.identity(ModDim)
    #theta = np.linalg.solve(A_p_kI,g)
    #print(theta)
    return  np.linalg.solve(A_p_kI,g)

def mkDiagCov(vrs):
    return np.diag(vrs)


def PotExAD_Scomp(a, gvec, sig, ModDm, z, kap, obsIndx, mode = None):
    return (2*sig**2)**(-1)*(z - getThA(ModDm, a, gvec,kap)[obsIndx])**2


def PotExAD(a, gvec, sig, ModDm, z, kap, dataDim, mode = None):
    #print((z - getThA(ModDm, a, gvec, kap))[0:dataDim])
    #print(norm((z - getThA(ModDm, a, gvec, kap))[0:dataDim]))
    return (2*sig**2)**(-1)*(norm((z - getThA(ModDm, a, gvec, kap))[0:dataDim]))**2

#This is function expects data_comp to be a vector of zeros and ones.  The ones pick out the observed directions

def PotExAD_comp(a, gvec, sig, ModDm, z, kap, dataDim, data_comp, mode = None):
    return (2*sig**2)**(-1)*(norm((z - getThA(ModDm, a, gvec, kap))*data_comp ))**2

def PotExAD_slice(a, gvec, sig, ModDm, z, kap, data_st, data_end, mode = None):
    # Negative log-likelihood for a contiguous observation window of theta.
    return (2*sig**2)**(-1)*(norm((z - getThA(ModDm, a, gvec, kap)[data_st:data_end]) ))**2


#This function expects a list of observation directions

def PotExAD_proj(Aprm, gvec, sig, ModDm, z, kap, obsdir, mode = None):
    thA = getThA(ModDm, Aprm, gvec, kap)
    thAProj = np.array([v @ thA for v in obsdir])
    return (2*sig**2)**(-1)*(norm(z -thAProj ))**2



def block_skew_Astar(d, omegas):
    """
    Build block diagonal skew matrix:
      diag( [ [0,w1;-w1,0], ..., [0,wn;-wn,0], [0] ] )
    Requires d = 2n+1, len(omegas)=n
    """
    assert d % 2 == 1, "d must be odd"
    n = (d - 1)//2
    assert len(omegas) == n

    A = np.zeros((d, d), dtype=float)
    for k, w in enumerate(omegas):
        i = 2*k
        A[i, i+1]   =  w
        A[i+1, i]   = -w
    return A


def make_omegas_power(d, beta=0.5, c=1.0, offset=1.0):
    """
    # Power-law decay for nearest-neighbor couplings.
    # omegas[i] = c * (offset + i)^(-beta), i = 0..d-2
    Useful for nearest-neighbor chain A[i,i+1]=omegas[i].
    """
    i = np.arange(d - 1, dtype=float)
    return c * (offset + i)**(-beta)

def make_Astar_nn(d, omegas):
    """
    # Nearest-neighbor skew-symmetric A* with A[i,i+1]=omega[i].
        A[i, i+1] = omegas[i]
        A[i+1, i] = -omegas[i]
    Requires len(omegas) == d-1.
    """
    omegas = np.asarray(omegas, dtype=float)
    if d < 2:
        raise ValueError("d must be >= 2")
    if omegas.shape != (d - 1,):
        raise ValueError(f"omegas must have shape ({d-1},), got {omegas.shape}")

    A = np.zeros((d, d), dtype=float)
    idx = np.arange(d - 1)
    A[idx, idx + 1] = omegas
    A[idx + 1, idx] = -omegas
    return A

def make_Astar_banded(d, bandwidth=2, omega_r_fn=None, rng=None):
    """
    General banded skew-symmetric A* with bandwidth 'bandwidth' >= 1.

    For each offset r=1..bandwidth and i=0..d-r-1:
        A[i, i+r] = omega_r(i, r)
        A[i+r, i] = -A[i, i+r]

    Parameters
    ----------
    d : int
    bandwidth : int
        Maximum |i-j| allowed for nonzero couplings.
    omega_r_fn : callable or None
        Function omega_r_fn(i, r) -> coupling strength.
        If None, defaults to omega_r_fn(i,r)= (1+r)^(-1) * (1+i)^(-1/2).
    rng : np.random.Generator or None
        Optional RNG if your omega_r_fn uses randomness.

    Returns
    -------
    A : (d,d) ndarray, skew-symmetric
    """
    if d < 2:
        raise ValueError("d must be >= 2")
    if bandwidth < 1:
        raise ValueError("bandwidth must be >= 1")

    if omega_r_fn is None:
        def omega_r_fn(i, r):
            return (1.0 + r)**(-1.0) * (1.0 + i)**(-0.5)

    A = np.zeros((d, d), dtype=float)
    for r in range(1, bandwidth + 1):
        for i in range(0, d - r):
            w = omega_r_fn(i, r) if rng is None else omega_r_fn(i, r, rng)
            A[i, i + r] = w
            A[i + r, i] = -w
    return A


def observe_single_component(theta_vec, idx):
    return float(theta_vec[idx])

def Pot_single_obs(Apar, gvec, sig, ModDm, z, kap, obs_idx):
    """
        Negative log-likelihood (up to additive constant) for one scalar observation:
      z = theta(A)[obs_idx] + noise, noise ~ N(0, sig^2)
    """
    th = getThA(ModDm, Apar, gvec, kap)
    pred = th[obs_idx]
    r = z - pred
    return 0.5*(r/sig)**2



def Find_AD_Match(numParm, modDm, gMean, gCov, Cov, kap, tol, transFn, ObsOp, max_tries, jobID, seed, stop_event, progress):
    """
    Search function to find problem parameters Obs(theta(q1)) = Obs(theta(q2)) for problem set up C
    These are run as tasks below within parallelization
    """
    rng = np.random.default_rng(seed)
    last_update = 0 
    
    curErr = np.inf
    mnA = np.zeros(numParm)
    CovInv = np.linalg.inv(Cov)
    numtries = 1

    bestMatch = {
        "err": np.inf,
        "A err": np.inf,
        "exp percent": 0,
        "A": None,
        "g": None,
        "tA": None,
        "tries": 0
    }

    numtries = 0
    bestErr = np.inf
    curErr = np.inf
    nmexpans = 0
    
    for t in range(1, max_tries + 1):
        if (t % 1024) == 0 and stop_event.is_set():
            return False, bestMatch
        Acur = rng.multivariate_normal(mnA, Cov)
        Gcur = rng.multivariate_normal(gMean, gCov)
        tAcur = transFn(Acur)
        curAerr = np.linalg.norm(ObsOp(getThA(modDm, tAcur, Gcur, kap)) - ObsOp(getThA(modDm, Acur, Gcur, kap))) 
        transInBulk = max(0,np.dot(CovInv @ tAcur, tAcur) -np.dot(CovInv @ Acur, Acur))
        nmexpans += int(transInBulk > 0)
        curErr = curAerr + transInBulk
        numtries = numtries+ 1
        if curErr <= bestErr:
            bestErr = curErr
            bestMatch["err"] = bestErr
            bestMatch["A err"] = curAerr
            bestMatch["exp percent"]= nmexpans/t
            bestMatch["A"] = Acur
            bestMatch["g"] = Gcur
            bestMatch["tA"] = tAcur
            bestMatch["tries"] = numtries
            if curErr <= tol:
                return True, bestMatch  # success, with the successful sample (also best)
    if (t - last_update) >= 4096:
        progress[jobID] = (t, bestErr)
        last_update = t
    
    return False, bestMatch

def Find_AD_Match_Parallel(numParm, modDm, gMean, gCov, Cov, kap, tol, transFn, ObsOp, maxTries_round=500000, max_rounds=48):
    n_workers = mp.cpu_count()
    total_tries = 0
    global_best = None
    secs_update = 15.0
    

    with mp.Manager() as manager:
        stop_event = manager.Event()   # picklable proxy
        progress = manager.dict()   # jobID -> (tries, bestErr)
        
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for r in range(max_rounds):
                # Create unique seeds per worker per round
                base_seed = np.random.SeedSequence().entropy
                
                jobs = []
                for jobID in range(n_workers):
                    seed = base_seed + 10_000 * r + jobID
                    jobs.append((numParm, modDm, gMean, gCov, Cov, kap, tol, transFn, ObsOp,maxTries_round,jobID, seed, stop_event,progress))

                futures = [pool.submit(Find_AD_Match, *job) for job in jobs]

                last_print = time.time()
                while futures:
                    
                    now = time.time()
                    if now - last_print >= secs_update:
                        items = list(progress.items())
                        tot = sum(t for _, (t, _) in items) if items else 0
                        best_seen = min((e for _, (_, e) in items), default=float("inf"))
                        gb = global_best["err"] if global_best else float("inf")
                        print(f"\rround {r+1}/{max_rounds} | total iters ~{tot:,} | best(worker) {best_seen:.3e} | best(global) {gb:.3e}", end="")
                        last_print = now

                    done = [f for f in futures if f.done()]
                    if not done:
                        time.sleep(0.05)
                        continue

                    for fut in done:
                        futures.remove(fut)
                        success, best = fut.result()
                        total_tries += best["tries"]

                        if (global_best is None) or (best["err"] < global_best["err"]):
                            global_best = best

                        if success:
                            stop_event.set()
                            for f in futures:
                                f.cancel()
                            print()  # newline after the \r line
                            return True, global_best, total_tries

            print()  # newline
    print("No suitable parameters found!! :(")
    return False, (global_best or {}), total_tries



# Some functions transition functions for the AD Match Utilities

def rot_A(A, numParm, lam = .9):
    return lam * rndm_orth_matrix(numParm) @ A


def getComps(theta, compArray):
    return compArray * theta


# Some features to search for a good value of A






        


