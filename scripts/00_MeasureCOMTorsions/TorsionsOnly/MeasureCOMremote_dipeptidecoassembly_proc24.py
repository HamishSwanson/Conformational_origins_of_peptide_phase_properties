import os
import math
import sys
import numpy as np
import h5py
import logging
import re
import MDAnalysis as mda
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
    def __init__(self, gro_file, species, molNum):
        self.file_path = os.getcwd()
        self.gro_file  = gro_file
        self.species   = species
        self.molNum    = molNum
        # self.hdf5_file = f'{self.file_path}/{self.species}/data_{self.molNum}_{self.species}.h5'
        self.hdf5_file = f'{gro_file.parent}/data_{self.molNum}_{self.species}.h5'

    def save_to_hdf5(self, dihedrals):
        with h5py.File(self.hdf5_file, 'w') as hf:
            hf.create_dataset('dihedrals', data=dihedrals)
        logging.info(f"Data saved to {self.hdf5_file}")
        return self.hdf5_file

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
    def __init__(self, gro_file, xtc_file, peptide_length, molNum):
        self.file_path      = os.getcwd()
        self.gro_file       = gro_file
        self.xtc_file       = xtc_file
        self.peptide_length = peptide_length
        self.molNum         = molNum
        self.traj           = mda.Universe(gro_file, xtc_file,in_memory=False) ## save in memory to step more quickly
        self.step           = 1

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
    def calculate_dihedrals(backbone, side_chains, residue_pairs):
        contact_list = []
        for j, pair in enumerate(residue_pairs):
            res1 = pair[0]
            res2 = pair[1]
            BackCOM1 = backbone.select_atoms(f'resid {res1}').center_of_mass()
            BackCOM2 = backbone.select_atoms(f'resid {res2}').center_of_mass()
            SideCOM1 = side_chains[res1-1].center_of_mass()
            SideCOM2 = side_chains[res2-1].center_of_mass()
            ##############################
            ab0      = SideCOM1 - BackCOM1
            bc0      = BackCOM1 - BackCOM2
            cd0      = BackCOM2 - SideCOM2
            dihed    = math.degrees(dihedral(ab0, bc0, cd0))
            ##############################
            contact_list.append(dihed)
        return contact_list

    def write_positions(self, ts, backbone, side_chains):
        location = self.gro_file.parent
        if ts == 0 or ts == (len(self.traj.trajectory)-1):
            for index in range(1, self.peptide_length + 1):
                with XYZWriter(f'{location}/Back{index}_{ts}.xyz') as W:
                    W.write(backbone.select_atoms(f'resid {index}'))
                with XYZWriter(f'{location}/Side{index}_{ts}.xyz') as W:
                    W.write(side_chains[index - 1])

    def process_trajectory_frames(self):
        protein           = self.traj.select_atoms('protein')
        protein_positions = self.traj.select_atoms('protein').positions
        molecule_length   = int(protein_positions.shape[0]/self.molNum)
        residue_pairs     = self.residue_pair_list(self.peptide_length)
        #########################################################
        num_frames        = len(self.traj.trajectory[::self.step])
        num_torsions      = len(residue_pairs)
        dihedral_array    = np.zeros((num_frames, num_torsions, self.molNum))
        #########################################################
        for frame_idx, ts in enumerate(range(0, num_frames, self.step)):
            self.traj.trajectory[frame_idx]
            index1        = 0
            index2        = molecule_length - 1
            dihedrals     = []
            dihedral_temp = np.zeros((num_torsions, self.molNum)) 
            for mol in range(self.molNum):
                backbone, sidechains  = self.process_atom(protein.select_atoms(f'index {index1}:{index2}'))
                dihedral_angles       = self.calculate_dihedrals(backbone, sidechains, residue_pairs)
                dihedral_temp[:, mol] = dihedral_angles
                self.write_positions(ts, backbone, sidechains) 
                index1 += molecule_length
                index2 += molecule_length
            dihedral_array[frame_idx, :, :] = dihedral_temp
        return dihedral_array

class DihedralAnalysis:
    def __init__(self, path=Path('.'), log_level=logging.DEBUG):
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
        print(self.gro_files)


    def process_file_pair(self, file_pair):
        graph_folder        = self.setup_graph_folders()
        gro_file, xtc_file  = file_pair
        self.molNum         = int((str(gro_file).split('/')[-1]).split('_')[0])
        self.species        = (str(gro_file).split('/')[-1]).split('_')[1]
        self.peptide_length = int(len(self.species))
        gro_dir             = gro_file.parent
        h5_files            = list(gro_dir.glob('*.h5'))
        if not h5_files:
            processor          = TrajectoryProcessor(gro_file, xtc_file, self.peptide_length, self.molNum)
            dihedral_arrays    = processor.process_trajectory_frames()
            saver              = DataSaver(gro_file, self.species, self.molNum)
            hdf5_file          = saver.save_to_hdf5(dihedral_arrays)
            h5_files           = list(gro_dir.glob('*.h5'))
            print(f'Done: {gro_dir}')
        else:
            print(f'Already Done: {gro_dir}')
            pass
        plot                   = Plotter(self.species, self.molNum, graph_folder, gro_file, self.peptide_length)
        plot.plot_data(h5_files[0])

    def parallel_run(self):
        print(f"Length of the gro_files: {len(self.gro_files)}")
        print(f"Length of the xtc_files: {len(self.xtc_files)}")
        if len(self.gro_files) != len(self.xtc_files):
            logging.error(f"gro_files ({len(self.gro_files)}) and xtc files ({len(self.xtc_files)}) do not match! Please check their numbers.")
            raise RuntimeError("Number of .gro and .xtc files do not match")
        file_pairs    = list(zip(self.gro_files, self.xtc_files))
        num_processes = 24
        with Pool(processes=num_processes) as pool:
            pool.map(self.process_file_pair, file_pairs)
        logging.debug("Trajectory files loaded successfully.")

    def main(self):
        self.init_data()
        self.parallel_run()

## copied from /XX/XX/XX/XX/XX/XX/XX (16/07/2025)
## revised version made on (17/07/2025) -- better handling of torsion measurement in calculate_dihedrals() detailed breakdown of function in benchling sheet (week commencing 14-07-25)
## mv MeasureCOMremote.py MeasureCOMremote_dipeptidecoassembly.py -- renamed on (17/07/2025) to better express specific function

if __name__ == "__main__":
    DihedralAnalysis.setup_logging(logging.DEBUG)
    path = os.getcwd() 
    path_object = Path(path)
    analysis    = DihedralAnalysis(path_object)
    analysis.main()
    sys.exit()