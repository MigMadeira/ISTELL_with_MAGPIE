Optimizations of the ISTELL equilibria with the grid "ISTELL_1cm_cubes_nodiagnostics_v3.focus".
The parameters for GPMO were:
# Set some hyperparameters for the optimization
algorithm = 'ArbVec'  # Algorithm to use
nAdjacent = 1  # How many magnets to consider "adjacent" to one another
nHistory = 295 # How often to save the algorithm progress
thresh_angle = np.pi # The angle between two "adjacent" dipoles such that they should be removed
max_nMagnets = 5900
nBacktracking = 200
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = 5900 # Maximum number of GPMO iterations to run
kwargs['nhistory'] = nHistory