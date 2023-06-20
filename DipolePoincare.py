import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField, ToroidalField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import PermanentMagnetGrid, create_equally_spaced_curves, curves_to_vtk
from simsopt.field import Current, ScaledCurrent, coils_via_symmetries, compute_fieldlines
from simsopt.solve import relax_and_split, GPMO 
from simsopt._core import Optimizable
import pickle
import time
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec
from simsopt import load
from math import sqrt,ceil
from mpi4py import MPI
from simsopt.util import MpiPartition
mpi = MpiPartition()
comm = MPI.COMM_WORLD

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
surface_flag = 'wout'
input_name = 'wout_ISTTOK_final.nc'
coordinate_flag = 'toroidal'


# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

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
OUT_DIR = '../../../Transfer/'
#os.makedirs(OUT_DIR, exist_ok=True)

# Files for the desired initial coils, magnet grid and magnetizations:
coilfile = (Path(__file__).parent / "inputs" / "biot_savart_opt_ISTELL.json").resolve()
famus_file = "../../tests/test_files/Poincare_inputs/ISTELL.focus"
dipole_file = "../../../Transfer/best_result_m=21328.txt"

# Get the Biot Savart field from the coils:
bs = load(coilfile)

# Plot initial Bnormal on plasma surface from the coils
#make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

# Set up correct Bnormal from the coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Initialize the permanent magnet class
pm_opt = PermanentMagnetGrid(
    s, Bn=Bnormal, 
    filename=surface_filename,
    coordinate_flag=coordinate_flag,
    famus_filename=famus_file
)


# Get the Biot Savart field from the magnets:
pm_opt.m = np.loadtxt(dipole_file)
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))

print("Total fB = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

bs_final = b_dipole + bs

# Do Poincaré plots from here

print("Obtaining VMEC final surfaces")
nfp = vmec_final.wout.nfp
nzeta = 4
nradius = 4
nfieldlines = nradius
zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
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
tmax_fl= 1000
tol_poincare=1e-16
def trace_fieldlines(bfield, R0, Z0):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol_poincare, comm=comm,
        phis=phis, stopping_criteria=[])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    #particles_to_vtk(fieldlines_tys, f'fieldlines_optimized_coils')
    return fieldlines_tys, fieldlines_phi_hits, phis

print("started tracing")
fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs_final, R0, Z0)


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
plt.savefig(OUT_DIR + f'poincare_ISTELL_time=1000_tol=1e-16_best.pdf', bbox_inches = 'tight', pad_inches = 0)
