import os,ast
import math
import sys
import numpy as np
import pandas as pd
import h5py
import logging
import re
import MDAnalysis as mda
from sklearn.metrics.pairwise import euclidean_distances
from MDAnalysis.coordinates.XYZ import XYZWriter
from MDAnalysis.lib.mdamath import dihedral
from multiprocessing import Pool, cpu_count
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.constants import Boltzmann, Avogadro, R
from scipy import signal
from scipy import ndimage

class DataSaver:
    def __init__(self,conf_data,analysis_file_name,label):
        self.conf_data          = conf_data
        self.analysis_file_name = analysis_file_name
        self.label              = label

    def save_to_hdf5(self):
        with h5py.File(self.analysis_file_name, 'w') as hf:
            hf.create_dataset(self.label, data=self.conf_data)

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

def FindMinima(x_grid,y_grid,z_grid):
    local_minima               = ndimage.minimum_filter(z_grid,size=3,mode='wrap') == z_grid ### returns array where min in square grid of 9 are returned for that neighbourhood | then boolean to find where this == its z_grid value
    labeled_minima, num_minima = ndimage.label(local_minima)                     ### returns labels for each minima (1,...,n) and number of minima
    sorted_minima              = np.column_stack(np.where(local_minima))         ### location of minima within the z_grid stacked into an array
    ################################################ 
    coord_x = []
    coord_y = []
    energy  = []
    for x_value,y_value in sorted_minima:
        coord_x.append(x_grid[x_value, y_value])
        coord_y.append(y_grid[x_value, y_value])
        energy.append(z_grid[x_value, y_value])
    df = pd.DataFrame({
        'Coord_X': coord_x,
        'Coord_Y': coord_y,
        'Energy (kJ/mol)': energy})
    minima_depth = 10
    df_sorted = df.sort_values(by='Energy (kJ/mol)', ascending=True)
    top_n     = df_sorted.head(minima_depth).to_numpy()
    num_rows  = top_n.shape[0]
    num_cols  = top_n.shape[1]
    if num_rows < minima_depth:
        padded_array = np.full((minima_depth, num_cols), np.nan)
        padded_array[:num_rows, :] = top_n
    else:
        padded_array = top_n
    np.set_printoptions(suppress=True, precision=5)
    print(padded_array)
    return padded_array

def probability_distribution(dihedral_flat,bond_flat):
    x             = dihedral_flat
    y             = bond_flat
    combined_rows = np.row_stack((x,y))
    xgrid         = 1
    ygrid         = 0.1
    x_dim         = np.arange(120,480,xgrid)                     ## update additional element (360,)
    y_dim         = np.arange(2,12+ygrid,ygrid)                  ## update additional element (700,) ## updated on July 26, 2026 - decrease lower bound of mesh                    
    x_grid,y_grid = np.meshgrid(x_dim, y_dim)
    grid          = np.vstack([x_grid.ravel(), y_grid.ravel()])
    kde_data      = gaussian_kde(combined_rows) ### scott
    ### for the all_400 dataset I find empirically that 1.3* Scott is good
    ### for the replicates dataset I used Scott and it works fine, so I will keep it as is 
    bandwidth_factor = (kde_data.factor) * 1.3
    if remote == True:
        bandwidth_factor = kde_data.factor
    bwlabel = f'{round(bandwidth_factor,3)}'
    print(f'Scott = {kde_data.factor}',f'Bandwidth = {bandwidth_factor}', f'BW = {bwlabel}')
    ###
    kde_data      = gaussian_kde(combined_rows,bw_method=bandwidth_factor)
    pdf           = kde_data(grid)
    pdf           = np.where(pdf==0,1e-100,pdf)     ## add tiny amount to remove pdf = 0
    energy        = (-R*298*np.log(pdf))/1000       ## U(a1,a2) = -kTln(P)
    energy        = energy-np.min(energy)           ## normalize for zero minima
    z_grid        = energy.reshape(x_grid.shape) 
    z_grid[z_grid >= 20] = np.nan
    padded_array  = FindMinima(x_grid,y_grid,z_grid) 
    stacked_array = np.stack((x_grid, y_grid, z_grid), axis=-1)
    min_index     = np.unravel_index(np.argmin(z_grid), z_grid.shape)
    x_value       = x_grid[min_index]
    y_value       = y_grid[min_index]
    z_value       = z_grid[min_index]
    #################################################### make FEL plot
    fig           = plt.figure()
    ax            = fig.add_subplot(111, projection='3d')
    surf          = ax.plot_surface(x_grid, y_grid, z_grid, cmap='Reds', edgecolor='k', linewidth=0.1, alpha=0.7)
    ax.grid(True, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("λ (°)", fontsize=12)
    ax.set_ylabel(r"γ (Å)", fontsize=12)
    ax.set_zlabel("Energy (kJ/mol)", fontsize=12)
    ax.view_init(elev=55, azim=-45) 
    ax.set_ylim(2,12)
    ax.set_zticks(np.arange(0,60,20))
    ax.set_zlim(0,40)
    ticks          = generate_circular_ticks()
    tick_positions = [tick if tick >= 120 else tick + 360 for tick in ticks]
    tick_labels    = [f'{tick}' for tick in ticks]
    plt.xticks(tick_positions, tick_labels,fontsize=10)
    ####################################################
    if remote:
        analysis_file_name = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{species}/Run1/{species}_fel_bw{bwlabel}.h5'
        conf_data          = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{species}/Run{run}/{species}_confs_bw{bwlabel}.csv'
        np.savetxt(conf_data,padded_array,delimiter=',') 
        print(analysis_file_name)
        saver     = DataSaver(z_grid,analysis_file_name,'FEL')
        hdf5_file = saver.save_to_hdf5()
        plt.savefig(f'graphs/{species}{run}_3d',transparent=True)
    else:
        analysis_file_name =  f'{Nter}X/{species}/{species}_fel_bw{bwlabel}.h5'
        np.savetxt(f'{Nter}X/{species}/{species}_confs_bw{bwlabel}.csv',padded_array,delimiter=',') 
        print(analysis_file_name)
        saver     = DataSaver(z_grid,analysis_file_name,'FEL')
        hdf5_file = saver.save_to_hdf5()
        plt.savefig(f'graphs/{species}_3d',transparent=True)
    plt.close()

def Plot2D(dihedral_flat,bond_flat):
    fig = plt.figure(figsize=(4,4))
    plt.scatter(dihedral_flat,bond_flat,s=1,edgecolor='black',linewidth=0.05)
    plt.xlabel("λ (°)", fontsize=12)
    plt.ylabel(r"γ (Å)", fontsize=12)
    plt.ylim(2,12)
    ####################################################
    ticks          = generate_circular_ticks()
    tick_positions = [tick if tick >= 120 else tick + 360 for tick in ticks]
    tick_labels    = [f'{tick}' for tick in ticks]
    plt.xticks(tick_positions, tick_labels,fontsize=10)
    ####################################################
    plt.tight_layout()
    if remote:
        plt.savefig(f'graphs/{species}{run}',transparent=True)
    else:
        plt.savefig(f'graphs/{species}',transparent=True)
    plt.close()

def plot_data(hdf5_dihedral,hdf5_bond):
    with h5py.File(hdf5_dihedral, 'r') as hf:
        dihedral_data = hf['dihedrals'][:]
    with h5py.File(hdf5_bond, 'r') as hf:
        bond_data     = hf['bond'][:]
    dihedrals     = np.where(dihedral_data < 120, dihedral_data + 360, dihedral_data)
    dihedral_flat = dihedrals.flatten()
    bond_flat     = bond_data.flatten()
    print('Length dihedrals:',len(dihedral_flat))
    print('Length bonds:',len(bond_flat))
    Plot2D(dihedral_flat,bond_flat)
    probability_distribution(dihedral_flat,bond_flat)

### VA/FF       {remote=False,test=True}
### all_400     {remote=False,test=False}
### all_conc    {remote=True,test=False} 

remote   = sys.argv[1] == "True"
test     = sys.argv[2] == "True"

print(remote,test)

if not remote:

    # test = False
    if test:

        sequence =['VA','FF']
        print('Running VA/FF AUC')

        for species in sequence:
            Nter = species[0]
            dihedrals = f'{Nter}X/{species}/data_1_{species}_dihedral.h5'
            bonds     = f'{Nter}X/{species}/data_1_{species}_bond.h5'
            plot_data(dihedrals,bonds)

    else:

        SINGLE_SINGLE  = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        SINGLE         = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        # test = True
        print('Running all AUC')

        if test: 
            SINGLE_SINGLE  = ['A'] 
            SINGLE         = ['A']

        for Nter in SINGLE_SINGLE:
            for Cter in SINGLE:
                species   = f'{Nter}{Cter}'
                print(species)
                dihedrals = f'{Nter}X/{species}/data_1_{species}_dihedral.h5'
                bonds     = f'{Nter}X/{species}/data_1_{species}_bond.h5'
                plot_data(dihedrals,bonds)

else:

    sequences = ['VA','VF','II','IL','LI','LL','QW','IF','LF','FF','WR','YR','FR','FL']
    sequences = ['YR','FR']
    runs      = ['1','2','3']
    # test      = False
    print('Running conc AUC')

    if test:
        sequences = ['FF']
        runs      = ['1']

    for species in sequences:
        for run in runs:
            dihedrals = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{species}/Run1/data_1_{species}_dihedral.h5'
            bonds     = f'path/to/00_MeasureCOMTorsions/TorsionsBonds/ConcentrationSeries/test_trajectory/1{species}/Run1/data_1_{species}_bond.h5' 
            plot_data(dihedrals,bonds)