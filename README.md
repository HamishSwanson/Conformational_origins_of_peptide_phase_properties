# Conformational origins of peptide phase properties


**Workflow for conformational Analysis**

The workflow used involves firstly calculating COM torsions (00_MeasureCOMTorsions) from a gro/xtc trajectory, this requires both files have the processed name with structure (`{no.peptides}_{sequence}_noPBC`). This step generates datafiles in h5 format containing either bond or bond/torsion data. Using this data the folding ratio can be calculated for the molecule based on the torsional data (01_EvaluateFoldingRatio). The dynamicity index calculation is compartmentalized into two parts using both the bond and torsion data. The first stage involves generating the FEL (02_ConstructFEL), then evaluating the surface occupancy and subsequently computing the dynamicity (03_EvaluateFEL). Further coide details are provided below. 

```
################## Directory Structure ##################

├── scripts
│   ├── 00_MeasureCOMTorsions
│   │   ├── TorsionsBonds
│   │   │   ├── All400
│   │   │   │   ├── config
│   │   │   │   └── MeasureCOMremote_dipeptidecoassembly_proc18_bond_angle.py
│   │   │   └── ConcentrationSeries
│   │   │       ├── config
│   │   │       ├── graphs
│   │   │       ├── logs
│   │   │       ├── test_trajectory
│   │   │       └── MeasureCOMremote_dipeptidecoassembly_proc16_bond_angle.py
│   │   └── TorsionsOnly
│   │       ├── config
│   │       │   ├── atoms2.config
│   │       │   ├── atoms3.config
│   │       │   ├── atoms4.config
│   │       │   └── atoms.config
│   │       └── MeasureCOMremote_dipeptidecoassembly_proc24.py
│   ├── 01_EvaluateFoldingRatio
│   │   ├── 00_all_dipeptides
│   │   │   ├── AX
│   │   │   │   └── AA
│   │   │   ├── GMMTrace
│   │   │   │   ├── 1AA_bic.png
│   │   │   │   └── 1AA_fold.png
│   │   │   └── FoldingPreference.py
│   │   ├── 01_concentration_series
│   │   │   ├── VF_1
│   │   │   │   ├── GMM-fitting
│   │   │   │   ├── GMMs
│   │   │   │   ├── VF_100.png
│   │   │   │   ├── VF_1_100_calculation.csv
│   │   │   │   ├── VF_1_18_calculation.csv
│   │   │   │   ├── VF_1_1_calculation.csv
│   │   │   │   ├── VF_1_35_calculation.csv
│   │   │   │   ├── VF_1_52_calculation.csv
│   │   │   │   ├── VF_1_70_calculation.csv
│   │   │   │   ├── VF_18.png
│   │   │   │   ├── VF_1.png
│   │   │   │   ├── VF_35.png
│   │   │   │   ├── VF_52.png
│   │   │   │   ├── VF_70.png
│   │   │   │   └── VF_lnKfold.csv
│   │   │   └── F3-P1_AutoGMM-1D_all_refined_specific.py
│   │   └── AX_example
│   │       └── AA
│   │           └── data_1_AA_dihedral.h5
│   ├── 02_ConstructFEL
│   │   ├── graphs
│   │   │   ├── FF1_3d.png
│   │   │   └── FF1.png
│   │   ├── PlotData.py
│   │   └── RunPlotData.py
│   └── 03_EvaluateFEL
│       ├── AnalyzeFELDipeptides.py
│       └── dipeptides_auc_conc.csv
├── test_trajectory
│   └── 1FF
│       └── Run1
│           ├── 1_FF_noPBC.gro
│           └── 1_FF_noPBC.xtc
└── README.md
...
```

**00_MeasureCOMTorsions**

/TorsionsBonds/All400/MeasureCOMremote_dipeptidecoassembly_proc18_bond_angle.py
/TorsionsBonds/ConcentrationSeries/MeasureCOMremote_dipeptidecoassembly_proc16_bond_angle.py
/TorsionsOnly/MeasureCOMremote_dipeptidecoassembly_proc24.py

These files calculate the center of mass (COM) positions for a peptide from a Gromacs MD trajectory. They assign atoms to the sidechain (SC) or backbone (BB) based on definitions within the local config file. All three versions of the script have subtle differences, mainly in how the front end functions to load trajectories for processing. Please note, the script will find and solve all trajectories in and below the directory at which it is pointed at. Those denoted 'TorsionsBonds' calculate both the torsions SC1-BB1-BB2-SC2 and the distance SC1-SC2 at the same time. Data inputs require a gro/xtc pair containing "noPBC" named with the sequence `{no.peptides}_{sequence}_noPBC`. 

Output files are in the compressed h5 format with the shape (no.frames,angle) or (no.frames,angle,no.molecule). Note, this code can run in parallel with the assigned no. of tasks through pool. We include a test gro/xtc pair for validation and testing of the code (test_trajectory/1FF/Run1/). Furthermore, to ensure the config assignment is correct (for transfer to different forcefields with non-charmm based atom naming schemes) we save xyz files with the naming structure `{Back/Side}{resID}_{frame}.xyz`. Note: feature is not included in MeasureCOMremote_dipeptidecoassembly_proc16_bond_angle.py. 

**01_EvaluateFoldingRatio**

/00_all_dipeptides/FoldingPreference.py<br>
/01_concentration_series/F3-P1_AutoGMM-1D_all_refined_specific.py

These files evaluate the folding preference based on the weighted GMM estimates. 

FoldingPreference.py is a later iteration of F3-P1_AutoGMM-1D_all_refined_specific.py written for the all_400 dataset analysis, its function can be tested for the h5 file for AA inculded within 01_EvaluateFoldingRatio/AX/AA. 

F3-P1_AutoGMM-1D_all_refined_specific.py is an earlier iteration of this code written for the concentration series results, here we show the results for the VF run 1 dataset. Due to repo size limitations these gro/xtc files are not included but the results can be found in the zenodo repo associated with this work. Note, we include a unit test for these two scripts based on the FF trajectory (test_trajectory/1FF/Run1/) to calculate the folding reference, these give the same result indicating coherent calculations between these versions.

**02_ConstructFEL**

/PlotData.py
/RunPlotData.py

The free energy landscapes (FELs) described within this work are calculated using this code. It is currently configured to execute on the FF trajectory (test_trajectory/1FF/Run1/) within the repo. It produces 1D/3D plots of the results within a subfolder (graphs). This code was written to be amenable to use on the 400 molecule and concentration series datasets and these selections are controlled by an a runfile also included within the directory.

**03_EvaluateFEL**

/AnalyzeFELDipeptides.py

Finally, this script calculates the dynamicity index based on the FEL from the previous step. It is currently configured to execute on the FF trajectory (test_trajectory/1FF/Run1/) though it has been written to apply to both the 400 molecule and concentration series datasets and this is controlled by execution selections by the user within the script. 