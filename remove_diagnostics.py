import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid, cylinder_to_vtk
from simsopt.field import DipoleField, ToroidalField
import simsoptpp as sopp
from simsopt.util.permanent_magnet_helper_functions import *


# Set some parameters
nphi = 64  # need to set this to 64 for a real run
ntheta = 64  # same as above
Plot = True

input_name = 'wout_ISTTOK_final.nc'
coordinate_flag = 'cartesian'
famus_filename = 'grids/ISTELL_aligned_axis/ISTELL_1cm_cubes_radial_extent=0505m_aligned_nfp=2_nPhi=4_full.focus'
algorithm = 'ArbVec'

# Make the output directory
OUT_DIR = './grids/ISTELL_with_diagnostics/' 
os.makedirs(OUT_DIR, exist_ok=True)

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = str(TEST_DIR/input_name)
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

#set a dummy B.n
net_poloidal_current_Amperes = 3.7713e+6
mu0 = 4 * np.pi * 1e-7
RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
bs = ToroidalField(R0=1, B0=RB)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)


#initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename) 

print('Number of available dipoles before diagnostic removal = ', pm_opt.ndipoles)

pm_opt.m = np.zeros(pm_opt.ndipoles*3)
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m,
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,)

b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K")

#create shapes
cylinder_list = [sopp.Cylinder(1, 0.2, 0, 3*np.pi/2, 0.3, 0.2, np.pi, 0), 
                 sopp.Cylinder(np.array([0, 1.2, 0]), 0.3, 0.2, np.pi/2, 0),
                 sopp.Cylinder(1, 0.2, np.pi/4, np.pi/4, 0.3, 0.2, 0, 0)]

pm_opt.remove_dipoles_inside_shapes(cylinder_list)

print('Number of available dipoles after diagnostic removal = ', pm_opt.ndipoles)

pm_opt.m = np.zeros(pm_opt.ndipoles*3)
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m,
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,)
b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K_after_diagnostic_removal")

if Plot:
    for i in range(len(cylinder_list)):
        cylinder_to_vtk(cylinder_list[i], f"Plots/diagnostics/diagnostic {i}")

# write solution to FAMUS-type file
pm_opt.write_to_famus(Path(OUT_DIR))