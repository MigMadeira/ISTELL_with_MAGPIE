import os
import pickle
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import Current, Coil
from simsopt.geo import CurveRZFourier, curves_to_vtk

c1 = CurveRZFourier(quadpoints=np.linspace(0,1,128), nfp = 1, stellsym = False, order=0)
c1.set("x0", 0.46 + 0.1275) #r0
c1.set("x1", 0.2025 + 0.155) #z0

c2 = CurveRZFourier(quadpoints=np.linspace(0,1,128), nfp = 1, stellsym = False, order=0)
c2.set("x0", 0.46 - 0.1275) #r0
c2.set("x1", 0.2025 + 0.155) #z0

c3 = CurveRZFourier(quadpoints=np.linspace(0,1,128), nfp = 1, stellsym = False, order=0)
c3.set("x0", 0.46 + 0.1275) #r0
c3.set("x1", -(0.2025 + 0.155)) #z0

c4 = CurveRZFourier(quadpoints=np.linspace(0,1,128), nfp = 1, stellsym = False, order=0)
c4.set("x0", 0.46 - 0.1275) #r0
c4.set("x1", -(0.2025 + 0.155)) #z0

curves_to_vtk([c1, c2, c3, c4], "quadripole_test")
