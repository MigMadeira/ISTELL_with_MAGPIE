import os
from simsopt.geo import SurfaceRZFourier
from simsopt.util.permanent_magnet_helper_functions import *


nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above

# Make the output directory
OUT_DIR = 'Plots/ISTELL_surfaces_plots/'
os.makedirs(OUT_DIR, exist_ok=True)

#create the outside boundary for the PMs (corresponds to the coil limit)
s_out = SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='full torus', nfp=2, stellsym=True)
s_out.set_rc( 0, 0, 0.52)
s_out.set_rc( 1, 0, 0.2025)
s_out.set_zs( 1, 0, 0.2025)
s_out.to_vtk(OUT_DIR + "surf_coil_full")

#create the outside boundary for the PMs (corresponds to the coil limit)
s_out = SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='half period', nfp=2, stellsym=True)
s_out.set_rc( 0, 0, 0.52)
s_out.set_rc( 1, 0, 0.2025)
s_out.set_zs( 1, 0, 0.2025)
s_out.to_vtk(OUT_DIR + "surf_coil")

#create the outside boundary for the PMs (corresponds to the coil limit if we align the coils with the VV)
s_out = SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='half period', nfp=2, stellsym=True)
s_out.set_rc( 0, 0, 0.46)
s_out.set_rc( 1, 0, 0.2025)
s_out.set_zs( 1, 0, 0.2025)
s_out.to_vtk(OUT_DIR + "surf_coil_aligned")

#create the outside boundary for the PMs (corresponds to the coil limit if we align the coils with the VV)
s_out = SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='full torus', nfp=2, stellsym=True)
s_out.set_rc( 0, 0, 0.46)
s_out.set_rc( 1, 0, 0.2025)
s_out.set_zs( 1, 0, 0.2025)
s_out.to_vtk(OUT_DIR + "surf_coil_aligned_full")

#create the inside boundary for the PMs (corresponds to the copper shell)
s_in =  SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='half period', nfp=2, stellsym=True)
s_in.set_rc(0, 0, 0.46)
s_in.set_rc(1, 0, 0.1275)
s_in.set_zs(1, 0, 0.1275)
s_in.to_vtk(OUT_DIR + "surf_copper_shell")

#create the inside boundary for the PMs (corresponds to the copper shell)
s_in =  SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='full torus', nfp=2, stellsym=True)
s_in.set_rc(0, 0, 0.46)
s_in.set_rc(1, 0, 0.1275)
s_in.set_zs(1, 0, 0.1275)
s_in.to_vtk(OUT_DIR + "surf_copper_shell_full")

#create the VV surface
s_in =  SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='half period', nfp=2, stellsym=True)
s_in.set_rc(0, 0, 0.46)
s_in.set_rc(1, 0, 0.085)
s_in.set_zs(1, 0, 0.085)
s_in.to_vtk(OUT_DIR + "surf_VV")

#create the VV surface
s_in =  SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='full torus', nfp=2, stellsym=True)
s_in.set_rc(0, 0, 0.46)
s_in.set_rc(1, 0, 0.085)
s_in.set_zs(1, 0, 0.085)
s_in.to_vtk(OUT_DIR + "surf_VV_full")