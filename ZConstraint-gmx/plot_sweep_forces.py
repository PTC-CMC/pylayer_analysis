import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

""" Plot the mean forces for a particular sweep
saved to a sweep folder """
parser = argparse.ArgumentParser()
parser.add_argument("--sweep", help="Designate sweep")

args = parser.parse_args()
curr_dir = os.getcwd()
os.chdir(os.path.join(curr_dir, "sweep{}".format(args.sweep)))

all_forces = [forcefile for forcefile in os.listdir() if 'meanforce' in forcefile]
z_windows = np.loadtxt("z_windows.out")
n_forces = len(all_forces)
mean_forces = []
for i in range(n_forces):
    meanforce = np.loadtxt("meanforce{}.dat".format(i))
    mean_forces.append(meanforce)
    
fig, ax = plt.subplots(1,1)
ax.plot(z_windows, mean_forces)
ax.set_xlabel("z coordinate (nm)")
ax.set_ylabel("Mean force (kJ/mol-nm)")
ax.set_title("sweep{} forces".format(args.sweep))
plt.savefig("sweep{}_forces.jpg".format(args.sweep), transparent=True)
plt.savefig("sweep{}_forces.svg".format(args.sweep), transparent=True)
