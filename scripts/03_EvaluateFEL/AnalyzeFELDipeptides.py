import os,sys,h5py
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.constants import Boltzmann, Avogadro, R

def open_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as hf:
        experiment_results = hf['FEL'][:]
    return experiment_results

def Evaluate(FEL_data):
    thermal_limit = []
    for thermal_threshold in grid_points:
        mask              = (FEL_data <= thermal_threshold)
        low_energy_grid   = np.where(mask, FEL_data, np.nan)
        total_grid_points = low_energy_grid.shape[0] * low_energy_grid.shape[1]
        not_nan_mask      = ~np.isnan(low_energy_grid)
        count             = (np.count_nonzero(not_nan_mask) / total_grid_points) * 100
        thermal_limit.append(count)
    return thermal_limit

def Plot_Thermal_Units(ax,pos):
        skips = [0,6]
        for level in range(7):
            pos1  = [RT*level,RT*level]
            y     = [-1,101]
            ax[pos].plot(pos1,y,linestyle='-',color='grey',alpha=0.6,linewidth=0.25)
            ##################### Add text label
            if level not in skips:
                x     = pos1[0]+0.2
                y_fix = 95
                ax[pos].text(x,y_fix,f'{level} RT',color='grey')
            #####################

def Manage_Analysis(sequences,runs,df_columns,time='',label='',):

        if label == 'dipeptides':
            empty_array = np.zeros((len(sequences),len(df_columns)), dtype=object)

        experiment_areas = []

        for k, mol in enumerate(sequences):

            # if remote:
            #     experiment_areas = [mol]

            experiment_areas = [mol]

            for run in runs:

                if label == 'dipeptides':

                    if remote:
                        filepath = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{mol}/Run{run}/{mol}_fel_bw0.215.h5'
                    else:
                        filepath = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{mol}/Run{run}/{mol[0]}X/{mol}/{mol}_fel_bw0.24.h5'

                if os.path.exists(filepath):
                    FEL_data      = open_hdf5(filepath)
                    thermal_limit = Evaluate(FEL_data)
                    function_area = np.column_stack((grid_points,thermal_limit))   
                    ############################ integrate 
                    x_array = function_area[:,0]
                    y_array = function_area[:,1]
                    area_under_curve = round(np.trapz(y_array,x_array,dx=dx),sigfig)
                    print(f'{mol}: {area_under_curve}')
                    ##################################################
                    experiment_areas.append(area_under_curve)
                else:
                    experiment_areas.append(np.nan)

            if remote:
                if len(experiment_areas) > 2: ## added for purposes of GitHub testing 
                    experiment_areas.append(round(np.nanmean(experiment_areas[1:]),sigfig))
                    experiment_areas.append(round(np.nanstd(experiment_areas[1:]),sigfig))
                else:
                    pass

            if label == 'dipeptides':
                print(experiment_areas)
                empty_array[k,:] = experiment_areas

        #########################################################
        if remote:
            df       = pd.DataFrame(empty_array,columns=df_columns)
            # plt_length = len(mols) * 0.8
            # fig,ax     = plt.subplots(figsize=(plt_length,3))
            # heights  = df['Mean']
            # y_errors = df['Std']
            # x        = np.arange(1,len(mols)+1,1)
            # bars     = plt.bar(x, heights, yerr=y_errors, capsize=5,edgecolor='black',alpha=0.5)
            # ax.set_ylabel(r'AUC ($\% \cdot \mathrm{kJ/mol}$)',fontsize=12)
            # ax.set_xlabel('Sequences',fontsize=12)
            # ax.set_xticks(x, mols) #,rotation=45)
            # plt.subplots_adjust(bottom=.2,top=0.98,right=0.98)
            # plt.show()
            ###########################################
            output_name = f'dipeptides_auc_conc'
            df.to_csv(output_name+'.csv', index=False)
            ###########################################
        else:
            df = pd.DataFrame(empty_array,columns=df_columns)
            df['Sequence'] = mols
            output_name = f'dipeptides_auc'
            df.to_csv(output_name+'.csv', index=False)
        #########################################################

remote = True

### constants corner ###
RT          = R*298/1000
dx          = 0.1
step        = dx * RT
grid_points = np.arange(0,RT*5+step,step)
########################
dx          = 0.25
step        = dx
grid_points = np.arange(0,12+step,step)
########################
sigfig = 3

if not remote:

    SINGLE  = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    runs    = ['1']
    test    = False

    if test:
        SINGLE  = ['A']

    def BuildDirs():
        mols = []
        for AA1 in SINGLE:
            for AA2 in SINGLE:
                dipeptide = f'{AA1}{AA2}'
                mols.append(dipeptide)
        return mols

    mols = BuildDirs()

    df_columns=['Sequence','AUC']

    Manage_Analysis(mols,runs,df_columns,time='500ns',label='dipeptides',)


else:

    mols = ['VA','II','IL','LI','VF','LL','IF','LF','WR','QW','FR','YR','FF','FL']
    runs = ['1','2','3']
    test = True
    if test:
        mols = ['FF']
        runs = ['1']
    df_columns=['Sequence','Run1']
    Manage_Analysis(mols,runs,df_columns,time='500ns',label='dipeptides',)




sys.exit()
