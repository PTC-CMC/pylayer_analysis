import os
import sys
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""Look for the max force in each sweep """
n_sweeps = len([filename for filename in os.listdir() if 'sweep' in filename and os.path.isdir(filename)])

# Loop through each sweep folder
# Look at each meanforce{}.dat 
# Find the greatest mean force
# Plot the absolute value of the mean force vs sweep number
current_dir = os.getcwd()
max_forces = []
for sweep in range(n_sweeps):
    os.chdir(os.path.join(current_dir, 'sweep{}'.format(sweep)))
    mean_force_files = [filename for filename in os.listdir() if 'meanforce' in filename]
    mean_forces = []
    for force_file in mean_force_files:
        mean_force = np.loadtxt(force_file)
        mean_forces.append(abs(mean_force))
    max_forces.append(max(mean_forces))
    os.chdir(current_dir)

fig, ax = plt.subplots(1,1)
ax.scatter(range(len(max_forces)),max_forces)
ax.set_title("Max force in each sweep")
x_ticks = [i for i in range(0, len(max_forces)+1,5)]
ax.set_xticks(x_ticks)
plt.savefig("max_forces.jpg",transparent=True)
plt.savefig("max_forces.svg",transparent=True)
print("Plots saved to max_forces.jpg and max_forces.svg")
