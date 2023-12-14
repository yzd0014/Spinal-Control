import numpy as np

# create training data
link_counts = 1

num_of_targets = 0
max_training_angle = 0.9
angle_interval = 0.1
max_cocontraction = 0.5
cocontraction_interval = 0.1
traning_samples = []


if link_counts == 2:
    for i in np.arange(-max_training_angle, max_training_angle+0.1, angle_interval):
        for j in np.arange(-max_training_angle, max_training_angle+0.1, angle_interval):
            for k in np.arange(0, max_cocontraction+0.1, cocontraction_interval):
                traning_samples.append(np.array([i, j, k]))
                num_of_targets += 1
elif link_counts == 1:
    for i in np.arange(0, max_training_angle+0.1, angle_interval):
        for k in np.arange(0, max_cocontraction+0.1, cocontraction_interval):
            traning_samples.append(np.array([i, k]))
            num_of_targets += 1

print(f"total number of training samples: {num_of_targets}")