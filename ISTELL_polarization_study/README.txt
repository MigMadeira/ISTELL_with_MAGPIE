Optimizations of the ISTELL equilibria with the grid "ISTELL_1cm_cubes_nospacing.focus".
The parameters for GPMO were:
# Set some hyperparameters for the optimization
algorithm = 'ArbVec'  # Algorithm to use
nAdjacent = 1  # How many magnets to consider "adjacent" to one another
nHistory = 620 # How often to save the algorithm progress
max_nMagnets = 8060
kwargs = initialize_default_kwargs('GPMO')
kwargs['nhistory'] = nHistory