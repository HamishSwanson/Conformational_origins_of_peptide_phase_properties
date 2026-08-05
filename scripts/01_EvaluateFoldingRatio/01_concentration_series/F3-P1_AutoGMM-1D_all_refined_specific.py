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

# Function to fit GMM and calculate AIC/BIC for a given k
def fit_gmm_and_get_scores(k, combined_results):
    gmm = GaussianMixture(n_components=k, random_state=31, covariance_type='full', max_iter=100, n_init=5, tol=0.0005, init_params='kmeans')
    gmm.fit(combined_results)
    return k, gmm.aic(combined_results), gmm.bic(combined_results)

# Main function to perform grid search and plot results
def Iterate_Clusters(combined_results,unit_test=False):
    results = Parallel(n_jobs=N_cpus)(delayed(fit_gmm_and_get_scores)(k, combined_results) for k in range(2, 16)) ## increase from 16 to 18 || CAUTION do not use too much CPU
    clusters, aic_scores, bic_scores = zip(*results)
    packaged_results = np.column_stack((clusters,bic_scores))
    if not unit_test:
        plt.plot(clusters, aic_scores, marker='o', linestyle='--', color='red', label='AIC')
        plt.plot(clusters, bic_scores, marker='o', linestyle='--', color='blue', label='BIC')
        plt.xlabel('Number of Components')
        plt.ylabel('Score')
        plt.title('AIC and BIC for Different Numbers of Components')
        plt.legend(loc='upper right') 
        plt.savefig(f'GMM-fitting/{measurement}_fitGMM')
        plt.close()
    return packaged_results

def plot_gmm_model(gmm, data, k, ax=None, colors=None):
    contents, bins,_  = plt.hist(data, bins=360, range=(120,480), density=True, color='blue', edgecolor='dimgrey', alpha=0.7)
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
    tick_labels = [f'{tick}°' for tick in circular_ticks]
    plt.ylim(0,0.02)
    plt.xticks(tick_positions, tick_labels)
    plt.legend(frameon=False)
    plt.xlabel(f'Angle (°)')
    plt.ylabel(f'Probability')
    plt.savefig(f'GMMs/{measurement}_GMM_{k}')
    plt.close()
    ########################################################################################

def measureGMM(dihedral_results,conc,unit_test=False):
    folding      = []
    cluster_used = []
    for cluster_value in range(2,16):
        gmm = GaussianMixture(n_components=cluster_value, random_state=31, covariance_type='full', max_iter=100, n_init=5, tol=0.0005, init_params='kmeans')
        gmm.fit(dihedral_results)
        if not unit_test:
            plot_gmm_model(gmm,dihedral_results,cluster_value,concentrations)
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
        folding.append(ln_kfold)
    return folding

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

def CompareToMulti(prospective_path,conc='1',unit_test=True):
        hdf5_dihedral  = prospective_path
        with h5py.File(hdf5_dihedral, 'r') as hf:
            dihedral_data = hf['dihedrals'][:]
        #####################################################################
        if conc == '1':
            n = 0 
        else:
            n = 8000
        print(f'Slice size: {n}')
        #####################################################################
        dihedrals         = np.where(dihedral_data < 120, dihedral_data + 360, dihedral_data) ## phase shift torsions
        dihedral_flat     = dihedrals[n:,:].flatten()                                       ## take final 25%
        combined_results  = dihedral_flat.reshape(-1,1)
        packaged_results  = Iterate_Clusters(combined_results,unit_test=True)           ## return (cluster,bic_score)
        ln_kfold          = measureGMM(combined_results,conc,unit_test=True)            ## return ln_kfold for each cluster
        ##################################################################### 
        packaged_results  = np.column_stack((packaged_results,ln_kfold))                ## return (cluster,bic_score,ln_kfold)
        packaged_results  = packaged_results[~np.isnan(packaged_results).any(axis=1)]   ## drop out rows with np.nan to avoid skewing weighting | return boolean array, where only NOT np.nan = True
        ##################################################################### 
        bic_weights       = EstimateWeights(packaged_results[:,1])                      ## estimate weights via bic
        packaged_results  = packaged_results[~np.isnan(packaged_results).any(axis=1)]   ## drop out instances where np.nan is returned when weights cannot be calculated (due to only open or closed)
        ##################################################################### 
        if not len(packaged_results) > 0:
            bic_ln_kfold     = np.nan
            bic_weights      = np.nan
        else:
            bic_ln_kfold = np.sum(bic_weights*packaged_results[:,2])                   ## calculate bic_lnKfold
        print(round(bic_ln_kfold,3))

molecules   = ['VA','II','IL','LI','LL','VF','IF','LF','FL','FF','QW','WR','YR','FR']
runs        = ['1','2','3']
N_cpus      = 8 
root        = os.getcwd()
test        = True
unit_test   = True

if test:
   molecules = ['VF']
   runs      = ['1','2','3'] 

if unit_test:
    species          = 'FF'
    prospective_path = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{species}/Run1/data_1_{species}_dihedral.h5'
    CompareToMulti(prospective_path,conc='1',unit_test=True)
    sys.exit()


for each, species in enumerate(molecules):
    for that, run in enumerate(runs):
        concentrations        = ['1','9','18','26','35','52','70','100']
        expts_concentrations  = []
        equilibria_bic        = []
        No_clusters           = []
        folder_path           = f'{species}_{run}'
        ##########################################
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        if not os.path.exists(f'{folder_path}/GMM-fitting'):
            os.mkdir(f'{folder_path}/GMM-fitting')
        if not os.path.exists(f'{folder_path}/GMMs'):
            os.mkdir(f'{folder_path}/GMMs')
        ##########################################
        os.chdir(folder_path)
        print(f'{folder_path}', flush=True)
        for j, conc in enumerate(concentrations):
            prospective_path = f'path/to/concseries/data/in/zenodo/dataset/{species}/{conc}{species}/Run{run}/data_{conc}_{species}.h5'
            test_file        = f'{folder_path}/{folder_path}_{conc}.csv'
            measurement      = f'{folder_path}_{conc}'
            print(prospective_path, flush=True)
            if os.path.exists(prospective_path) and not os.path.exists(test_file):
                concentration  = round(((int(conc)/6.02214129e23)/3.43e-22),3)
                print(f'\nMeasure: {conc} [{concentration} mol/L]', flush=True)
                ##################################################################### define data file path & read
                hdf5_dihedral  = prospective_path
                with h5py.File(hdf5_dihedral, 'r') as hf:
                    dihedral_data = hf['dihedrals'][:]
                #####################################################################
                if conc == '1':
                    n = 0 
                else:
                    # n = 7500
                    n = 8000
                print(f'Slice size: {n}')
                #####################################################################
                dihedrals         = np.where(dihedral_data < 120, dihedral_data + 360, dihedral_data) ## phase shift torsions
                dihedral_flat     = dihedrals[n:,:,:].flatten()                                       ## take final 25%
                combined_results  = dihedral_flat.reshape(-1,1)
                packaged_results  = Iterate_Clusters(combined_results,unit_test=False)          ## return (cluster,bic_score)
                ln_kfold          = measureGMM(combined_results,conc,unit_test=False)           ## return ln_kfold for each cluster
                packaged_results  = np.column_stack((packaged_results,ln_kfold))                ## return (cluster,bic_score,ln_kfold)
                packaged_results  = packaged_results[~np.isnan(packaged_results).any(axis=1)]   ## drop out rows with np.nan to avoid skewing weighting | return boolean array, where only NOT np.nan = True
                bic_weights       = EstimateWeights(packaged_results[:,1])                      ## estimate weights via bic
                ##################################################################### 
                if not len(packaged_results) > 0:
                    bic_ln_kfold     = np.nan
                    bic_weights      = np.nan
                else:
                    bic_ln_kfold = np.sum(bic_weights*packaged_results[:,2])                   ## calculate bic_lnKfold
                    print(bic_ln_kfold)
                    print(f'Weighted Average (BIC): {bic_ln_kfold}', flush=True)
                    plt.scatter(0, bic_ln_kfold, marker='o', edgecolors='black', color='blue', label='Weighted Average (BIC)')
                    plt.plot(packaged_results[:,0],packaged_results[:,2],marker='o',markeredgecolor='blue',markerfacecolor='yellow',color='blue',linestyle='--',label='lnKfold Estimates')
                    plt.ylabel('Calculated lnKfold')
                    plt.xlabel('No. Cluster')
                    plt.legend(frameon=False)
                    plt.ylim(-4,5)
                    plt.savefig(f'{species}_{conc}')
                    plt.close()
                expts_concentrations.append(concentration)
                equilibria_bic.append(bic_ln_kfold)
                #################################################################
                if not len(packaged_results) > 0:
                    packaged_results = np.full(4, np.nan)
                    packaged_results = packaged_results.reshape(1, 4)
                else:
                    packaged_results = np.column_stack((packaged_results,bic_weights)) 
                ##############################################################################################################
                calculation_results = pd.DataFrame(packaged_results,columns=['Clusters','BIC_scores','lnKfold','BIC_weights'])
                calculation_results.to_csv(f'{folder_path}_{conc}_calculation.csv')
            else:
                print(f'Done or Doesnt exist! {test_file}', flush=True)
        #######################################################################################
        results                          = pd.DataFrame(columns=['Concentration (mol/L)','bic_lnKfold'])
        results['Concentration (mol/L)'] = expts_concentrations
        results['bic_lnKfold']           = equilibria_bic
        results.to_csv(f'{species}_lnKfold.csv')
        #######################################################################################
        os.chdir(root)
    else:
        print(f'Done {species}', flush=True)
        pass
    os.chdir(root)

sys.exit()
