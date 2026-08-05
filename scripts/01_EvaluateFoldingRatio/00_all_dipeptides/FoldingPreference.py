import os,sys,math

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import h5py
import MDAnalysis as mda
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from scipy.constants import Boltzmann, Avogadro, R
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from scipy import ndimage
from PIL import Image

def generate_circular_ticks(start=180, step=90, max_angle=360):
    ticks = []
    current_angle = start
    while current_angle < max_angle + step:
        ticks.append(current_angle)
        current_angle += step
    if current_angle >= max_angle:
        for angle in range(0, 121, step):
            ticks.append(angle)
    return ticks
circular_ticks = generate_circular_ticks()

def CalculateFold(gmm,cluster_value):
    combined_results         = np.column_stack((gmm.means_,gmm.weights_))
    closed_states            = combined_results[(combined_results[:, 0] >= 270) & (combined_results[:, 0] < 450)]
    open_states              = combined_results[(combined_results[:, 0] < 270) | (combined_results[:, 0] >= 450)]
    if len(closed_states) != 0 and len(open_states) != 0:
        Z_open                   = np.sum(open_states[:,1])         ## sum of open weights
        Z_closed                 = np.sum(closed_states[:,1])       ## sum of closed weights
        kfold                    = Z_closed/Z_open                  ## ratio gives equilibrium
        ln_kfold                 = np.log(kfold)                    ## take logarithm
    else:
        ln_kfold = np.nan
    return ln_kfold

def fit_gmm_and_get_scores(k, dihedral_results,species,label='',unit_test=False):
    gmm = GaussianMixture(n_components=k, random_state=31, covariance_type='full', max_iter=100, n_init=5, tol=0.0005, init_params='kmeans')
    if label == 'rapid':
        gmm.fit(dihedral_results)
        fold = CalculateFold(gmm,k)
        if not unit_test:
            plot_gmm_model(gmm, dihedral_results, k, species,ax=None, colors=None)
        return k, gmm.bic(dihedral_results),fold
    if label == 'measured':
        return gmm       

def Iterate_Clusters(dihedral_results,species,unit_test=False):
    results = Parallel(n_jobs=8)(delayed(fit_gmm_and_get_scores)(k, dihedral_results,species,label='rapid',unit_test=unit_test) for k in range(2, N_cluster)) ## increase from 16 to 18 || CAUTION do not use too much CPU
    clusters, bic_scores, folding = zip(*results)
    packaged_results     = np.column_stack((clusters,bic_scores,folding))
    return packaged_results

def plot_gmm_model(gmm, dihedral_results, k, species, ax=None, colors=None):
    contents, bins,_  = plt.hist(dihedral_results, bins=360, range=(120,480), density=True, color='lime', edgecolor='none', alpha=0.5)
    ############################################################################# GPT Derived
    x_original  = np.linspace(120,480,len(contents)).reshape(-1, 1)     # x-positions for probability estimate
    logprob     = gmm.score_samples(x_original)                         # estimate log-likelihood of each sample  
    pdf         = np.exp(logprob)                                       # estimate probability by taking exp() of log-likelihood
    plt.plot(x_original, pdf, '-b', label='Combined Model', linewidth=2)
    for i in range(k):                              # Plot individual Gaussian components
        mean   = gmm.means_[i]                      
        cov    = gmm.covariances_[i]
        weight = gmm.weights_[i]
        component_pdf = (weight *
            1 / (np.sqrt(2 * np.pi * cov)) *        # Correct normalization factor
            np.exp(-0.5 * (x_original - mean) ** 2 / cov)
        )
        plt.plot(x_original, component_pdf, '--', label=f'Gaussian {i + 1}')
    #############################################################################
    y=[0.0]*len(gmm.means_)
    plt.scatter(gmm.means_,y,color='r',marker='D')
    tick_positions = [tick if tick >= 120 else tick + 360 for tick in circular_ticks]  # Adjust positions
    tick_labels = [f'{tick}' for tick in circular_ticks]
    plt.ylim(0,0.02)
    plt.xticks(tick_positions, tick_labels)
    plt.legend(frameon=False)
    plt.xlabel(f'Angle (°)')
    plt.ylabel(f'Probability')
    plt.savefig(f'path/to/01_EvaluateFoldingRatio/00_all_dipeptides/{species[0]}X/{species}/GMM-fitting/{species}_{conc}_GMM_{k}',dpi=100,transparent=True)
    plt.close()
    ########################################################################################

def EstimateWeights(bic_values):
    if len(bic_values) > 0:
        min_bic      = np.min(bic_values)
        weights      = np.exp(-0.5 * (bic_values - min_bic))
        weights      = weights / np.sum(weights)
        ####
        valid        = ~np.isnan(weights)
        weights      = weights[valid]
        norm_weights = weights / np.sum(weights)
    else:
        norm_weights = np.full(len(range(2, N_cluster)), np.nan)
    return norm_weights

def BuildTestList(SINGLE,SINGLE_LETTERS):
    all_400 = []
    for AA1 in SINGLE:
        if not os.path.exists(f'{AA1}X'):
            os.mkdir(f'{AA1}X')
        for counter, AA2 in enumerate(SINGLE_LETTERS):
            sequence       = f"{AA1}{AA2}"
            all_400.append(sequence)
    return all_400

def ReturnTorsions(hdf5_dihedral,segment):
    hdf5_dihedral  = prospective_path
    with h5py.File(hdf5_dihedral, 'r') as hf:
        dihedral_data = hf[ 'dihedrals'][:]
    dihedrals         = np.where(dihedral_data < 120, dihedral_data + 360, dihedral_data)
    dihedral_flat     = dihedrals[segment:,:].flatten()
    dihedral_results  = dihedral_flat.reshape(-1,1)
    return dihedral_results

def PlotOPT(clusters, bic_scores,bic_weights):
    fig, ax1 = plt.subplots(figsize=(4,4))
    ax2 = ax1.twinx()
    ax2.plot(clusters, bic_weights, marker='o', linestyle='--', color='red', label='Model Weight',markersize=4)
    ax2.set_ylabel('BIC Weights')    # Correct method
    ax2.set_yticks(np.arange(0,1.2,0.2))
    ax1.plot(clusters, bic_scores, marker='o', linestyle='--', color='blue', label='BIC',markersize=4)
    ax1.set_xticks(np.arange(2,18,2))
    ax1.set_xlim(0,16)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Score')
    plt.title(f'{conc}{species} - BIC scores',fontsize=8)
    plt.legend(loc='best',frameon=False,fontsize=6) 
    plt.tight_layout()
    plt.savefig(f'../GMMTrace/{conc}{species}_bic',dpi=400,transparent=True)
    plt.close()

def PlotBIC(packaged_results):
    fig, ax1         = plt.subplots(figsize=(4,4))
    clusters    = packaged_results[:,0]
    fold_values = packaged_results[:,2]
    ax1.scatter(1, bic_ln_kfold, marker='^', edgecolors='black', color='blue', label='Weighted Average (BIC)')
    ax1.plot(clusters,fold_values,marker='o',markeredgecolor='blue',markerfacecolor='yellow',color='blue',linestyle='--',label='Fold Estimates')
    ax1.set_xticks(np.arange(2,18,2))
    ax1.set_xlim(0,16)
    ax1.set_title(f'{conc}{species} - Fold Estimates',fontsize=8)
    ax1.set_ylabel('Calculated lnKfold')
    ax1.set_xlabel('No. Cluster')
    ax1.legend(frameon=False)
    ax1.set_ylim(-4,5)
    plt.legend(frameon=False)
    fig.tight_layout()
    plt.savefig(f'../GMMTrace/{conc}{species}_fold',dpi=400,transparent=True)
    plt.close()

def CompareToMulti(prospective_path,conc='1',unit_test=True):
    dihedral_results  = ReturnTorsions(prospective_path,0)
    packaged_results  = Iterate_Clusters(dihedral_results,species,unit_test=True)   ## np.column_stack((clusters,bic_scores,fold))
    packaged_results  = packaged_results[~np.isnan(packaged_results).any(axis=1)]   ## drop out rows with np.nan as these are non-viable GMMs
    bic_weights       = EstimateWeights(packaged_results[:,1])                      ## estimate weights using BIC
    ##############################################################################################################
    if not len(packaged_results) > 0:                              ## set to zero if the calculation fails
        bic_ln_kfold     = np.nan                                  ##
        bic_weights      = np.nan                                  ## 
    else:
        bic_ln_kfold = np.sum(bic_weights*packaged_results[:,2])   ## if values exist calculated weighted estimate of Chi
    print(round(bic_ln_kfold,3))

### Procedure ###
## (1) ReturnTorsions(hdf5_dihedral,segment_def.get(conc)) --> dihedral_results => return phase shifted lambda
## (2) Iterate_Clusters(dihedral_results)                  --> packaged_results => np.column_stack((clusters,bic_scores,fold)) --> rapid parallel calculation
## (4) packaged_results  = np.column_stack((packaged_results,ln_kfold))              --> clean out failed estimates 
## (5) packaged_results  = packaged_results[~np.isnan(packaged_results).any(axis=1)] --> clean out failed estimates
## (6) EstimateWeights(packaged_results[:,1])                                        --> calculate model weights for viable GMMs
## (7) packaged_results  = packaged_results[~np.isnan(packaged_results).any(axis=1)] --> clean out any failed estimates
## (8) if method works --> bic_ln_kfold = np.sum(bic_weights*packaged_results[:,2])  --> evaluated weighted fold value
## (9) save weights    --> packaged_results = np.column_stack((packaged_results,bic_weights)) --> save this as a *csv for the conc and pass foward the bic_lnKfold value

SINGLE         = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
SINGLE_LETTERS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
root           = os.getcwd()
N_cluster      = 16
segment_def    = {'1':0,'70':8000}
test           = True
unit_test      = False

if test:
    SINGLE         = ['A']
    SINGLE_LETTERS = ['A']
all_400            = BuildTestList(SINGLE,SINGLE_LETTERS)

if unit_test:
    species          = 'FF'
    prospective_path = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{species}/Run1/data_1_{species}_dihedral.h5'
    CompareToMulti(prospective_path,conc='1',unit_test=True)
    sys.exit()

for each, species in enumerate(all_400):
    print(f'{species[0]}X', flush=True)
    os.chdir(f'{species[0]}X')
    os.system('pwd')
    concentrations                      = ['1']
    expts_concentrations,equilibria_bic = [],[]
    os.makedirs(species, exist_ok=True)
    os.makedirs(f'{species}/GMM-fitting', exist_ok=True)
    for j, conc in enumerate(concentrations):
        if conc == '1':
            prospective_path = f'path/to/01_EvaluateFoldingRatio/{species[0]}X_example/{species}/data_1_{species}_dihedral.h5'
        else:
            pass
        test_file      = f'{species}/{species}_{conc}_calculation.csv'
        concentration  = round(((int(conc)/6.02214129e23)/3.43e-22),3)
        print(prospective_path)
        if not os.path.exists(test_file):
            ##########################################################################
            dihedral_results  = ReturnTorsions(prospective_path,segment_def.get(conc))
            packaged_results  = Iterate_Clusters(dihedral_results,species)                  ## np.column_stack((clusters,bic_scores,fold))
            packaged_results  = packaged_results[~np.isnan(packaged_results).any(axis=1)]   ## drop out rows with np.nan as these are non-viable GMMs
            bic_weights       = EstimateWeights(packaged_results[:,1])                      ## estimate weights using BIC
            ##############################################################################################################
            if not len(packaged_results) > 0:                              ## set to zero if the calculation fails
                bic_ln_kfold     = np.nan                                  ##
                bic_weights      = np.nan                                  ## 
            else:
                bic_ln_kfold = np.sum(bic_weights*packaged_results[:,2])   ## if values exist calculated weighted estimate of Chi
                PlotOPT(packaged_results[:,0],packaged_results[:,1],bic_weights)
                PlotBIC(packaged_results)                                  ## plot how these compare between cluster numbers
            ##############################################################################################################
            if not len(packaged_results) > 0:
                packaged_results = np.full(4, np.nan)
                packaged_results = packaged_results.reshape(1, 4)
            else:
                packaged_results = np.column_stack((packaged_results,bic_weights)) 
            ############################################################################################################## save estimate for selected concentration
            calculation_results = pd.DataFrame(packaged_results,columns=['Clusters','BIC_scores','lnKfold','BIC_weights'])
            calculation_results.to_csv(f'{species}/{species}_{conc}_calculation.csv')
            ############################################################################################################## pass foward to make compiled results csv --> f'{species}_lnKfold.csv'
            expts_concentrations.append(concentration)
            equilibria_bic.append(bic_ln_kfold)
            pass
        else:
            print(f'Done or Doesnt exist! {test_file}', flush=True)
    ############################################################################################################## define compiled results csv
    results                          = pd.DataFrame(columns=['Concentration (mol/L)','bic_lnKfold'])
    results['Concentration (mol/L)'] = expts_concentrations
    results['bic_lnKfold']           = equilibria_bic
    results.to_csv(f'{species}/{species}_lnKfold.csv')
    ##############################################################################################################
    os.chdir(root)

sys.exit()