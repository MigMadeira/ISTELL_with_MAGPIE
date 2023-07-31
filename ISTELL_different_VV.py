import os
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import PermanentMagnetGrid, create_equally_spaced_curves, curves_to_vtk, Surface, cylinder_to_vtk
from simsopt.field import Current, DipoleField, coils_via_symmetries
import time
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.util import FocusData
from simsopt.util import discretize_polarizations, polarization_axes
import simsoptpp as sopp

t_start = time.time()

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
input_name = 'wout_ISTTOK_final_rescaled.nc'
coordinate_flag = 'cartesian'
famus_filename = 'grids/ISTELL_diff_VV/ISTELL_different_VV.focus'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
OUT_DIR = 'ISTELL_different_VV/'
os.makedirs(OUT_DIR, exist_ok=True)

#setting radius for the circular coils
vmec = Vmec(TEST_DIR / input_name)

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5
# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 5, multiply by 2*ncoils to get the total number of coils.)
ncoils = int(24/(2*s.nfp))

# Major radius for the initial circular coils:
R0 = 0.52

# Minor radius for the initial circular coils:
R1 = 0.2025

#Initialize the coils 
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(1.0) * 48e3 for i in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

# fix all the coil shapes so only the currents are optimized
for i in range(ncoils):
    base_curves[i].fix_all()
    base_currents[i].fix_all()

# Initialize the coil curves and save the data to vtk
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")

bs = BiotSavart(coils)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

# check after-optimization average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
bs.save(OUT_DIR + f"biot_savart_opt.json")

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

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, pol_vectors=pol_vectors) 

print('Number of available dipoles = ', pm_opt.ndipoles)

pm_opt.m = np.zeros(pm_opt.ndipoles*3)
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m,
                       nfp=1, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,stellsym=False)

b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K")

#create the outside boundary for the PMs
s_out = SurfaceRZFourier.from_wout(surface_filename, nphi = nphi, ntheta = ntheta, range='half period')
s_out.extend_via_normal(0.01)
s_out.to_vtk(OUT_DIR + "surf_out")

pm_opt.remove_magnets_inside_surface(s_out)

# remove any dipoles where the diagnostic ports should be
cylinder_list = [sopp.Cylinder(0.52+0.035, 0.03, 2*np.pi/12, np.pi/2, 0.21, 0.0575, 2*np.pi/12, 0),
                 sopp.Cylinder(0.52, 0.03, 2*np.pi/12, 0, 0.21, 0.035, 2*np.pi/12, np.pi/2),
                 sopp.Cylinder(0.52+0.035, 0.03, 2*np.pi/12, 3*np.pi/2, 0.21, 0.035, 2*np.pi/12, np.pi),
                 sopp.Cylinder(0.52-0.035, 0.03, 4*np.pi/12, np.pi/2, 0.21, 0.035, 4*np.pi/12, 0),
                 sopp.Cylinder(0.52, 0.03, 4*np.pi/12, np.pi/9, 0.21, 0.0575, 4*np.pi/12, np.pi/2)]
pm_opt.remove_dipoles_inside_shapes(cylinder_list)

b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m,
                       nfp=1, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,stellsym=False)

b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K_after_cylinder_removal")

print('Number of available dipoles after diagnostic port and toroidal removal = ', pm_opt.ndipoles)

for i in range(len(cylinder_list)):
    cylinder_to_vtk(cylinder_list[i], OUT_DIR + f"diagnostics/diagnostic {i}")

pm_opt.write_to_famus(Path(OUT_DIR))
