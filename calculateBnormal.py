import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import PermanentMagnetGrid
from simsopt.field import compute_fieldlines
import pickle
import time
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec
from simsopt import load
from math import sqrt,ceil
from mpi4py import MPI
from simsopt.util import MpiPartition, FocusData, discretize_polarizations, polarization_axes
mpi = MpiPartition()
comm = MPI.COMM_WORLD

# Set some parameters
comm = None
nphi = 64*4 # need to set this to 64 for a real run
ntheta = 64 # same as above
input_name = 'wout_ISTELL_final.nc'
coordinate_flag = 'cartesian'


# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="full torus", nphi=nphi, ntheta=ntheta)


qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

vmec_final = Vmec(TEST_DIR / input_name)
ntheta_VMEC = 200

# Make the output directory
OUT_DIR = './Poincare_plots/Data/'
os.makedirs(OUT_DIR, exist_ok=True)

# Files for the desired initial coils, magnet grid and magnetizations:
coilfile = "./ISTELL_aligned_axis/2cm_diag_backtracking/biot_savart_initial.json"
famus_filename = "./ISTELL_aligned_axis/2cm_diag_backtracking/SIMSOPT_dipole_solution.focus"
dipole_file = "./ISTELL_aligned_axis/2cm_diag_backtracking/best_result_m=9516.txt"

# Get the Biot Savart field from the coils:
bs = load(coilfile)

# Set up correct Bnormal from the coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

#load focus file with the grid info
mag_data = FocusData(famus_filename)

# Determine the allowable polarization types and reject the negatives
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

# Optionally add additional types of allowed orientations
PM4Stell_orientations = True
full_orientations = False
if PM4Stell_orientations:
    pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
    ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
    pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
    pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
    pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fe_ftri))

    pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
    ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
    pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
    pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
    pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fc_ftri))
    
    if full_orientations:
        pol_axes_corner, pol_type_corner = polarization_axes(['corner'])
        ntype_corner = int(len(pol_type_corner)/2)
        pol_axes_corner = pol_axes_corner[:ntype_corner, :]
        pol_type_corner = pol_type_corner[:ntype_corner] + 1
        pol_axes = np.concatenate((pol_axes, pol_axes_corner), axis=0)
        pol_type = np.concatenate((pol_type, pol_type_corner))
        
        pol_axes_edge, pol_type_edge = polarization_axes(['edge'])
        ntype_edge = int(len(pol_type_edge)/2)
        pol_axes_edge = pol_axes_edge[:ntype_edge, :]
        pol_type_edge = pol_type_edge[:ntype_edge] + 1
        pol_axes = np.concatenate((pol_axes, pol_axes_edge), axis=0)
        pol_type = np.concatenate((pol_type, pol_type_edge))
        

#setup the polarization vectors from the magnet data in the focus file
ophi = np.arctan2(mag_data.oy, mag_data.ox) 
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z
print('pol_vectors_shape = ', pol_vectors.shape)

s.nfp = 1
s.stellsym = False
# Initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, 
                                                  famus_filename, pol_vectors=pol_vectors)

# Get the Biot Savart field from the magnets:
pm_opt.m = np.loadtxt(dipole_file)
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m.reshape(pm_opt.ndipoles * 3),
                       nfp=1, stellsym=False, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))

Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles

Bnormal_avg=0

for i in range(len(Bnormal_total)):
    for j in range(len(Bnormal_total[i])):
        Bnormal_avg+=abs(Bnormal_total[i][j])
        
dim1,dim2=np.shape(Bnormal_total)
Bnormal_avg = Bnormal_avg/(dim1*dim2)    
print(Bnormal_avg)