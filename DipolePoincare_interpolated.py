import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier, CurveRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import PermanentMagnetGrid
from simsopt.field import compute_fieldlines, InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion
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

vmec_final = Vmec(TEST_DIR / input_name)
ntheta_VMEC = 100

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
OUT_DIR = './Poincare_plots/interpolated/'
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
nfp = s.nfp
stellsym = True
mpol=5
ntor=5

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

#interpolating the magnetic field
n = 20
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, n*2)
nfieldlines = 30
degree = 4

if stellsym:
    zrange = (0, np.max(zs), n//2)
else:
    zrange = (-np.max(zs), np.max(zs), n)

s_levelset = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                      range="full torus", nphi=64, ntheta=24)
s_levelset.fit_to_curve(ma, 0.40, flip_theta=False)

s_levelset.to_vtk(OUT_DIR + 'surface')
sc_fieldline = SurfaceClassifier(s_levelset, h=0.03, p=2)
sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

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

# free up memory
import gc
del b_dipole
del bs
del bs_final
del pm_opt
del pol_vectors, pol_axes, pol_type
del ophi
del mag_data
gc.collect()

# Do Poincaré plots from here

print("Obtaining VMEC final surfaces")
nfp = vmec_final.wout.nfp
nzeta = 4
nradius = 4
nfieldlines = nradius
zeta = np.linspace(0,np.pi/nfp,num=nzeta,endpoint=False)
theta = np.linspace(0,2*np.pi,num=ntheta_VMEC)
iradii = np.linspace(0,vmec_final.wout.ns-1,num=nfieldlines).round()
iradii = [int(i) for i in iradii]
R_final = np.zeros((nzeta,nradius,ntheta_VMEC))
Z_final = np.zeros((nzeta,nradius,ntheta_VMEC))
for itheta in range(ntheta_VMEC):
    for izeta in range(nzeta):
        for iradius in range(nradius):
            for imode, xnn in enumerate(vmec_final.wout.xn):
                angle = vmec_final.wout.xm[imode]*theta[itheta] - xnn*zeta[izeta]
                R_final[izeta,iradius,itheta] += vmec_final.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
                Z_final[izeta,iradius,itheta] += vmec_final.wout.zmns[imode, iradii[iradius]]*np.sin(angle)
R0 = R_final[0,:,0]
Z0 = Z_final[0,:,0]

print("Finished VMEC")
from simsopt.field import particles_to_vtk
tmax_fl= 10000
tol_poincare=1e-12
def trace_fieldlines(bfield, R0, Z0):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol_poincare, comm=comm,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    #particles_to_vtk(fieldlines_tys, f'fieldlines_optimized_coils')
    return fieldlines_tys, fieldlines_phi_hits, phis

print("started tracing")
fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bsh, R0, Z0)


print('Creating Poincare plot R, Z')
r = []
z = []
for izeta in range(len(phis)):
    r_2D = []
    z_2D = []
    for iradius in range(len(fieldlines_phi_hits)):
        lost = fieldlines_phi_hits[iradius][-1, 1] < 0
        data_this_phi = fieldlines_phi_hits[iradius][np.where(fieldlines_phi_hits[iradius][:, 1] == izeta)[0], :]
        if data_this_phi.size == 0:
            print(f'No Poincare data for iradius={iradius} and izeta={izeta}')
            continue
        r_2D.append(np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2))
        z_2D.append(data_this_phi[:, 4])
    r.append(r_2D)
    z.append(z_2D)
r = np.array(r, dtype=object)
z = np.array(z, dtype=object)
print('Plotting Poincare plot')
nrowcol = ceil(sqrt(len(phis)))
fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 8))
for i in range(len(phis)):
    row = i//nrowcol
    col = i % nrowcol
    axs[row, col].set_title(f"$\\phi={phis[i]/np.pi:.2f}\\pi$", loc='right', y=0.0, fontsize=10)
    axs[row, col].set_xlabel("$R$", fontsize=14)
    axs[row, col].set_ylabel("$Z$", fontsize=14)
    axs[row, col].set_aspect('equal')
    axs[row, col].tick_params(direction="in")
    for j in range(nfieldlines):
        if j== 0 and i == 0:
            legend1 = 'Poincare'
            legend3 = 'VMEC'
        else:
            legend1 = legend2 = legend3 = '_nolegend_'
       
        axs[row, col].plot(R_final[i,j,:], Z_final[i,j,:], '-', linewidth=1.2, c='k', label = legend3)
        try: axs[row, col].scatter(r[i][j], z[i][j], marker='o', s=1.3, linewidths=1.3, c='b', label = legend1)
        except Exception as e: print(e, i, j)
        # if j == 0: axs[row, col].legend(loc="upper right")
# plt.legend(bbox_to_anchor=(0.1, 0.9 ))
leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)
plt.tight_layout()

plt.savefig(OUT_DIR + f'poincare_ISTELL_PM4STELL_time={tmax_fl}_tol={tol_poincare}.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.savefig(OUT_DIR + f'poincare_ISTELL_PM4STELL_time={tmax_fl}_tol={tol_poincare}.png', bbox_inches = 'tight', pad_inches = 0)
