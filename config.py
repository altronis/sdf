num_samples = 500000  # Number of points to sample around the point cloud

use_dropout = False  # Whether to use dropout after FC layers
batch_size = 256
lr = 1e-4
epochs = 5
delta = 0.1  # Delta when calculating clamped L1 loss

log_interval = 100
res = 64  # Rendering resolution
