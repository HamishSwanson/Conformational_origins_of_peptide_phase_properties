import os
import math
import sys
import numpy as np
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

class ConfigReader:
    def __init__(self, peptide_length):
        self.file_path = os.path.join(os.getcwd(), "config", f"atoms{peptide_length}.config")
        self.peptide_length = peptide_length

    def read_config(self):
        if not os.path.exists(self.file_path):
            logging.error(f"Config file {self.file_path} does not exist.")
            raise FileNotFoundError(f"Config file {self.file_path} does not exist.")
        with open(self.file_path, "r") as f:
            return f.read().split("\n")

class DataSaver:
    def __init__(self, gro_file, species, molNum,local_directory):
        self.file_path = os.getcwd()
        self.gro_file  = gro_file
        self.species   = species
        self.molNum    = molNum
        self.local_directory = local_directory
        self.hdf5_file_dihedral  = f'{self.local_directory}/data_{self.molNum}_{self.species}_dihedral.h5'
        self.hdf5_file_sidechain = f'{self.local_directory}/data_{self.molNum}_{self.species}_bond.h5'

    def save_to_hdf5(self,input,var_name=None):

        if var_name == "dihedral_array":
            hdf5_file = self.hdf5_file_dihedral
            print('Saving Dihedrals!')
            print(input,hdf5_file,'\n')
            with h5py.File(hdf5_file, 'w') as hf:
                hf.create_dataset('dihedrals', data=input)
            logging.info(f"Data saved to {self.hdf5_file_dihedral}")
        
        if var_name == "sc_array":
            hdf5_file = self.hdf5_file_sidechain
            print('Saving Bonds!')
            print(input,hdf5_file,'\n')
            with h5py.File(hdf5_file, 'w') as hf:
                hf.create_dataset('bond', data=input)
            logging.info(f"Data saved to {self.hdf5_file_sidechain}")
            
        return hdf5_file

class Plotter:
    def __init__(self, species, molNum, path, gro_file, peptide_length):
        self.species        = species
        self.molNum         = molNum
        self.path           = path
        self.run            = gro_file.parent.name
        self.exptname       = gro_file.parent.parent.parent.name
        self.exptrun        = gro_file.parent.parent.name.replace('Run','')
        self.peptide_length = peptide_length

    def plot_data(self, hdf5_file):
        residue_pairs = TrajectoryProcessor.residue_pair_list(self.peptide_length)
        with h5py.File(hdf5_file, 'r') as hf:
            dihedral_data = hf['dihedrals'][:]  # Shape: (frames, num_torsions, molNum)
        num_frames, num_torsions, molNum = dihedral_data.shape
        ####################################### Determine subplot grid: max 3 rows
        nrows = min(3, num_torsions)
        ncols = math.ceil(num_torsions / nrows)
        #######################################
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20 * ncols, 9* nrows)) #, constrained_layout=True)
        # axes      = axes.flatten()  # to index subplots easily
        #######################################
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]  # wrap single Axes into a list
        #######################################
        for i in range(num_torsions):
            res_pair = residue_pairs[i]
            torsion_data = dihedral_data[:, i, :].flatten()
            torsion_data = np.where(torsion_data < 120, torsion_data + 360, torsion_data)
            bins = np.arange(120,481,1)
            axes[i].hist(torsion_data, bins=bins, density=True,color='lightcoral', edgecolor='black', alpha=0.7)
            axes[i].set_xticks(np.arange(120, 481, 60))
            axes[i].set_yticks(np.arange(0.0,0.03,0.005))    
            axes[i].set_title(f"Torsion {res_pair}", fontsize=24)
            axes[i].set_xlabel("Dihedral Angle (°)", fontsize=24)
            axes[i].set_ylabel("Density", fontsize=24)
            axes[i].tick_params(axis='both', labelsize=18)
        ######################################## Hide any extra subplots
        for j in range(num_torsions, len(axes)):
            fig.delaxes(axes[j])
        ########################################
        plt.savefig(f'{self.path}/{self.species}_{self.molNum}_{self.exptname}_{self.exptrun}.jpg')
        plt.close()

class TrajectoryProcessor:
    def __init__(self, gro_file, xtc_file, peptide_length, molNum,local_directory):
        self.file_path      = os.getcwd()
        self.gro_file       = gro_file
        self.xtc_file       = xtc_file
        self.peptide_length = peptide_length
        self.molNum         = molNum
        self.traj           = mda.Universe(gro_file, xtc_file,in_memory=True) ## save in memory to step more quickly
        self.step           = 1
        self.local_directory = local_directory

    def process_atom(self, molecule):
        config_reader = ConfigReader(self.peptide_length)
        content = config_reader.read_config()
        if len(content) < self.peptide_length:
            raise ValueError("Config file does not contain enough lines.")
        backbone    = molecule.select_atoms(content[0])
        side_chains = [molecule.select_atoms(content[i]) for i in range(1, self.peptide_length + 1)]
        return backbone, side_chains

    @staticmethod
    def residue_pair_list(peptide_length):
        ######################################## Determine all possible measurements of sidechains
        residue_pairs = []
        for i in range(1, peptide_length):
            residue_pairs.append([i, i+1])
        for i in range(1, peptide_length):    
            if i + 2 <= peptide_length:
                residue_pairs.append([i, i+2])
        ########################################
        return residue_pairs

    @staticmethod
    def calculate_dihedrals(backbone, peptide_length, side_chains):
        for i in range(1, peptide_length):
            BackCOM1 = backbone.select_atoms(f'resid {i}').center_of_mass()
            BackCOM2 = backbone.select_atoms(f'resid {i+1}').center_of_mass()
            SideCOM1 = side_chains[i-1].center_of_mass()
            SideCOM2 = side_chains[i].center_of_mass()
            ##############################
            ab0      = SideCOM1 - BackCOM1
            bc0      = BackCOM1 - BackCOM2
            cd0      = BackCOM2 - SideCOM2
            dihed    = math.degrees(dihedral(ab0, bc0, cd0))
            ##############################
            SideCOM1 = SideCOM1.reshape(1,-1)
            SideCOM2 = SideCOM2.reshape(1,-1)
            distance = euclidean_distances(SideCOM1,SideCOM2)[0][0]
            ##############################
        return dihed,distance #contact_list

    def write_positions(self, ts, backbone, side_chains, species):
        # location = self.gro_file.parent
        location = self.local_directory
        if ts == 0 or ts == (len(self.traj.trajectory)-1):
            for index in range(1, self.peptide_length + 1):
                with XYZWriter(f'{location}/Back{index}_{ts}.xyz') as W:
                    W.write(backbone.select_atoms(f'resid {index}'))
                with XYZWriter(f'{location}/Side{index}_{ts}.xyz') as W:
                    W.write(side_chains[index - 1])

    def process_trajectory_frames(self,species):
        protein           = self.traj.select_atoms('protein')
        protein_positions = self.traj.select_atoms('protein').positions
        molecule_length   = int(protein_positions.shape[0]/self.molNum)
        step              = self.step
        for ts in range(0, len(self.traj.trajectory), step):
            self.traj.trajectory[ts]
            index1 = 0
            index2 = molecule_length - 1
            dihedrals = []
            sc_distances = []
            for mol in range(self.molNum):
                backbone, sidechains = self.process_atom(protein.select_atoms(f'index {index1}:{index2}'))
                self.write_positions(ts, backbone, sidechains, species)
                dihedral_angle,bond_dist = self.calculate_dihedrals(backbone, self.peptide_length, sidechains)
                dihedrals.append(dihedral_angle)
                sc_distances.append(bond_dist)
                if mol == 0 or mol == self.molNum - 1:
                    logging.debug(f'Peptide: Processing molecule {mol + 1} in frame {ts}: {dihedral_angle}')
                index1 += molecule_length
                index2 += molecule_length
            if ts == 0:
                dihedral_array = np.array((dihedrals))
                sc_array       = np.array((sc_distances))
            else: 
                dihedral_array = np.vstack((dihedral_array,dihedrals))    
                sc_array       = np.vstack((sc_array,sc_distances))   
        return dihedral_array,sc_array

class DihedralAnalysis:
    def __init__(self,path=Path('.'), log_level=logging.DEBUG):
        self.file_path      = os.getcwd()
        self.file_name      = Path()
        self.species        = ""    ## default is unknown and inferred in process_file_pair() - will fail if not defined correctly
        self.molNum         = ""    ## default is unknown and inferred in process_file_pair() - will fail if not defined correctly
        self.peptide_length = ""    ## default is unknown and inferred in process_file_pair() - will fail if not defined correctly
        self.path           = path
        self.gro_files      = []
        self.xtc_files      = []
        self.setup_logging(log_level)

    @staticmethod
    def setup_logging(log_level):
        log_dir = 'logs'
        Path(log_dir).mkdir(exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s - %(process)d - %(filename)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.FileHandler(Path(log_dir) / 'app.log')])

    def setup_graph_folders(self):
        graph_folder = self.file_path + "/graphs"
        if os.path.isdir(graph_folder):
            return graph_folder
        else:
            os.mkdir(graph_folder)
            return graph_folder

    def init_data(self):
        gro_files = list(self.path.rglob('*.gro'))
        xtc_files = list(self.path.rglob('*.xtc')) # if not 
        self.file_name = gro_files[0].parent.parent.name
        self.gro_files = sorted([file for file in gro_files if 'noPBC' in str(file)])
        self.xtc_files = sorted([file for file in xtc_files if 'noPBC' in str(file)])
        # print(self.gro_files)


    def process_file_pair(self, file_pair):
        graph_folder        = self.setup_graph_folders()
        gro_file, xtc_file  = file_pair
        filename            = (str(gro_file).split('/')[-1])
        number_var, species_var, *_ = filename.split('_')
        self.molNum         = int(number_var)
        self.species        = species_var
        self.peptide_length = int(len(self.species))
        gro_dir             = gro_file.parent.parent
        subfolders          = Path(*gro_file.parts[-4:-1]) 
        local_directory     = Path.cwd() / subfolders
        subfolders          = Path(*gro_file.parts[-4:-1]) 
        local_dir           = Path.cwd() / subfolders
        if not os.path.exists(local_directory):
            local_directory.mkdir(parents=True, exist_ok=True)
        h5_files  = False
        if os.path.exists(f'{local_directory}/data_1_{self.species}_bond.h5') and os.path.exists(f'{local_directory}/data_1_{self.species}_dihedral.h5'):
            h5_files = True
        if not h5_files:
            processor       = TrajectoryProcessor(gro_file, xtc_file, self.peptide_length, self.molNum,local_directory)
            dihedral_array,sc_array = processor.process_trajectory_frames(self.species)
            saver           = DataSaver(gro_file, self.species, self.molNum,local_directory)
            hdf5_file       = saver.save_to_hdf5(sc_array,var_name="sc_array")
            hdf5_file       = saver.save_to_hdf5(dihedral_array,var_name="dihedral_array")
            print(f'Done: {gro_dir}')
        else:
            print(f'Already Done: {gro_dir}')
            pass

    def parallel_run(self):
        print(f"Length of the gro_files: {len(self.gro_files)}")
        print(f"Length of the xtc_files: {len(self.xtc_files)}")
        if len(self.gro_files) != len(self.xtc_files):
            logging.error(f"gro_files ({len(self.gro_files)}) and xtc files ({len(self.xtc_files)}) do not match! Please check their numbers.")
            raise RuntimeError("Number of .gro and .xtc files do not match")
        file_pairs = list(zip(self.gro_files, self.xtc_files))
        candidates = ['VA','VF','II','QW','IF','LF','FF','WR','IL','LI','LL','FL','YR','FR']
        filtered_file_pairs = [(gro, xtc) for gro, xtc in file_pairs if any(cand in str(gro) for cand in candidates)]
        num_processes       = 16
        with Pool(processes=num_processes) as pool:
            pool.map(self.process_file_pair, filtered_file_pairs)
        logging.debug("Trajectory files loaded successfully.")

    def main(self):
        self.init_data()
        self.parallel_run()

## copied from /XX/XX/XX/XX/XX/XX/XX (16/07/2025)
## revised version made on (17/07/2025) -- better handling of torsion measurement in calculate_dihedrals() detailed breakdown of function in benchling sheet (week commencing 14-07-25)
## mv MeasureCOMremote.py MeasureCOMremote_dipeptidecoassembly.py -- renamed on (17/07/2025) to better express specific function

if __name__ == "__main__":
    DihedralAnalysis.setup_logging(logging.DEBUG)
    path        = f'path/to/test_trajectory/1FF/Run1/'
    path_object = Path(path)
    analysis    = DihedralAnalysis(path_object)
    analysis.main()
    sys.exit()