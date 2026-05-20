"""
Configuration parameters for NN training
"""

# Data selection
MASS_WINDOW = (2.9969 - 0.1, 2.9969 + 0.1)  # J/psi mass window [GeV]
VTX_PROB_MIN = 0.01
DR_MIN = 0.12
ETA_MAX = 1.5

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE_TRAIN = 4096
BATCH_SIZE_VAL = 8192

# Model architecture
N_FEATURES = 3
HIDDEN_DIMS = (64, 64, 32)

# Optimization
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Reference binning for 2D validation
PT_BINS = [6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 10.5, 11.0, 12.0, 20.0, 100.0]
DXY_BINS = [0.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 500.0]

# Fine binning for 1D projections
PT_BINS_1D = (0, 30, 60)      # (min, max, nbins)
DXY_BINS_1D = (0, 15, 30)     # (min, max, nbins)
