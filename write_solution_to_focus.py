import os
import pickle
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField, Current, coils_via_symmetries
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier, create_equally_spaced_curves, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util import FocusData, discretize_polarizations, polarization_axes
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec

t_start = time.time()

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
surface_flag = 'wout'
input_name = 'wout_ISTTOK_final.nc'
coordinate_flag = 'cartesian'
famus_filename = 'grids/ISTELL_1cm_cubes_nodiagnostics_v3.focus'
dipole_file = "./ISTELL_with_spacing/PM4STELL/best_result_m=52500.txt"

# Read in the plasma equilibrium file
TEST_DIR = Path(__file__).parent
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory
OUT_DIR = 'ISTELL_with_spacing/PM4STELL/'
os.makedirs(OUT_DIR, exist_ok=True)

#setting radius for the circular coils
vmec = Vmec(TEST_DIR / input_name)

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5
# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 5, multiply by 2*ncoils to get the total number of coils.)
ncoils = int(24/(2*s.nfp))

# Major radius for the initial circular coils:
R0 = vmec.wout.Rmajor_p

# Minor radius for the initial circular coils:
#R1 = vmec.wout.Aminor_p + poff + coff + 0.5
R1 = 0.28

total_current = Vmec(surface_filename).external_current() / (2 * s.nfp)
print("Total Current= ", total_current)
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(total_current / ncoils) for _ in range(ncoils-1)]
total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

# fix all the coil shapes so only the currents are optimized
for i in range(ncoils):
    base_curves[i].fix_all()

bs = BiotSavart(coils)
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
        pol_type_corner = pol_type_corner[:ntype_corner] + 3
        pol_axes = np.concatenate((pol_axes, pol_axes_corner), axis=0)
        pol_type = np.concatenate((pol_type, pol_type_corner))
        
        pol_axes_edge, pol_type_edge = polarization_axes(['edge'])
        ntype_edge = int(len(pol_type_edge)/2)
        pol_axes_edge = pol_axes_edge[:ntype_edge, :]
        pol_type_edge = pol_type_edge[:ntype_edge] + 4
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

#initialize the permanent magnet class an write to file
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, pol_vectors=pol_vectors) 
pm_opt.m = np.loadtxt(dipole_file)
pm_opt.write_to_famus(Path(OUT_DIR))

    
