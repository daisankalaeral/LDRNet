n_points = 100
if n_points > 4:
    size_per_border = int((n_points-4)/4)

reg_ratio = 50
beta = 0.01
gamma = 0.01
lr = 0.0003
batch_size = 128
num_workers = 4
valid_interval = 1
warmup_step = 7500