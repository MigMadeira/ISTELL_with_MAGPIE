import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier, CurveRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.field.biotsavart import BiotSavart
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data,\
    InterpolatedField
from simsopt.geo import PermanentMagnetGrid
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
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
input_name = 'wout_ISTTOK_final.nc'
coordinate_flag = 'cartesian'


# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

nfp = s.nfp

vmec_final = Vmec(TEST_DIR / input_name)
ntheta_VMEC = 200

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
OUT_DIR = './Poincare_plots/tracing/'
os.makedirs(OUT_DIR, exist_ok=True)

# Files for the desired initial coils, magnet grid and magnetizations:
coilfile = "./ISTELL_with_spacing/PM4STELL/biot_savart_opt.json"
famus_filename = "./grids/ISTELL_1cm_cubes_nodiagnostics_v3.focus"
dipole_file = "./ISTELL_with_spacing/PM4STELL/result_m=39060.txt"

# Get the Biot Savart field from the coils:
bs = load(coilfile)

# Plot initial Bnormal on plasma surface from the coils
#make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

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

# Initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, 
                                                  famus_filename, pol_vectors=pol_vectors)

# Get the Biot Savart field from the magnets:
pm_opt.m = np.loadtxt(dipole_file)
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m.reshape(pm_opt.ndipoles * 3),
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))

print("Total fB = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

bs_final = b_dipole + bs


# Get the magnetic axis

raxis_cc = [0.455004988086597, 0.0440156683282455, 0.00272439261412522, 
   0.000172253586663152, 1.0252868811645e-05, 5.67166789928362e-07, 
   4.45152913520489e-08, 1.03095879617727e-08, 7.56214434436398e-10, 
   -1.92098079138534e-10,0]
   
zaxis_cs = [-0, -0.0439018971482906, -0.00270493882880684, 
   -0.000172074499581258, -1.12324407227628e-05, -7.85113554306607e-07, 
   -6.18546981576889e-08, -1.15638995441075e-08, -3.24721336851853e-09, 
   -1.38710486183217e-08]

Nt_ma=10
ppp=10

numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
ma = CurveRZFourier(numpoints, Nt_ma, nfp, True)
ma.rc[:] = raxis_cc[0:(Nt_ma+1)]
ma.zs[:] = zaxis_cs[0:Nt_ma]
ma.x = ma.get_dofs()
    
# Do Poincaré plots from here

print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs_final.set_points(ma.gamma()).B(), axis=1)))
print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

s_levelset = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                      range="full torus", nphi=64, ntheta=24)
s_levelset.fit_to_curve(ma, 0.70, flip_theta=False)

s_levelset.to_vtk(OUT_DIR + 'surface')
sc_fieldline = SurfaceClassifier(s_levelset, h=0.03, p=2)
sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

nfieldlines = 30
degree = 4
tmax_fl = 400000

stellsym = True

def trace_fieldlines(bfield, label):
    t1 = time.time()
    R0 = np.linspace(ma.gamma()[0, 0], ma.gamma()[0, 0] + 0.14, nfieldlines)
    Z0 = [ma.gamma()[0, 2] for i in range(nfieldlines)]
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-7, comm=comm,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm is None or comm.rank == 0:
        particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)


# Bounds for the interpolated magnetic field chosen so that the surface is
# entirely contained in it
n = 20
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, n*2)
# exploit stellarator symmetry and only consider positive z values:
if stellsym:
    zrange = (0, np.max(zs), n//2)
else:
    zrange = (-np.max(zs), np.max(zs), n)

def skip(rs, phis, zs):
    # The RegularGrindInterpolant3D class allows us to specify a function that
    # is used in order to figure out which cells to be skipped.  Internally,
    # the class will evaluate this function on the nodes of the regular mesh,
    # and if *all* of the eight corners are outside the domain, then the cell
    # is skipped.  Since the surface may be curved in a way that for some
    # cells, all mesh nodes are outside the surface, but the surface still
    # intersects with a cell, we need to have a bit of buffer in the signed
    # distance (essentially blowing up the surface a bit), to avoid ignoring
    # cells that shouldn't be ignored
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
    print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip


print('Initializing InterpolatedField')
bsh = InterpolatedField(
    bs_final, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=stellsym, skip=skip
)
print('Done initializing InterpolatedField')

bsh.set_points(ma.gamma().reshape((-1, 3)))
bs.set_points(ma.gamma().reshape((-1, 3)))
Bh = bsh.B()
B = bs.B()
print("|B-Bh| on axis", np.sort(np.abs(B-Bh).flatten()))

print('Beginning field line tracing')
trace_fieldlines(bsh, 'ISTTOK_final_PM4STELL')