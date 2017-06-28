from SystemSetup import *
import numpy as np
from optparse import OptionParser
import os
import subprocess

# Run this after the permeability simulations have finished
# This means after stage 5 has finisehd running
# And all force files are still in xvg form


# sim_folder/sweep{}/sim{}
# Force files should be moved to sweep{}
n_sweeps = len([filename for filename in os.listdir() if 'sweep' in filename and os.path.isdir(filename)])

# Read in the forces files, splitting 
# Them into different force files
current_dir = os.getcwd()
for sweep in range(n_sweeps):
    for i in range(N_sims):
        filename = "Stage5_ZCon"+str(i)+"_pullf.xvg"
        os.chdir(os.path.join(current_dir, "sweep{}/Sim{}".format(sweep, i)))
        all_forces = np.loadtxt(filename, skiprows=30)
        for j in range(N_tracer):
            force_index = (N_tracer * j) + i
            np.savetxt(os.path.join(current_dir, "sweep{}/forceout{}.dat".format(sweep, force_index)), 
                    np.column_stack((all_forces[:, 0], all_forces[:, j+1])))





#for i in range(11):
#    p = subprocess.Popen("mkdir sweep{}".format(i), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    p.wait()
#    p = subprocess.Popen("cp z_windows.out sweep{}".format(i), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    p.wait()
#
#current_dir = os.getcwd()
#for i in range(N_sims):
#    filename = "Stage5_ZCon"+str(i)+"_pullf.xvg"
#    os.chdir(os.path.join(current_dir, "Sim{}".format(i)))
#    all_forces = np.loadtxt(filename, skiprows=30)
#    for j in range(N_tracer):
#        force_index = (N_tracer * i) + j
#        np.savetxt(os.path.join(current_dir, "sweep0/forceout{}.dat".format(force_index)), 
#                np.column_stack((all_forces[:, 0], all_forces[:, j+1])))

# Get all the other force files
#for h in range(10):
#    for i in range(N_sims):
#        filename = "#Stage5_ZCon" + str(i) + "_pullf.xvg." + str(h+1) + "#"
#        os.chdir(os.path.join(current_dir, "Sim{}".format(i)))
#        all_forces = np.loadtxt(filename, skiprows=30)
#        for j in range(N_tracer):
#            force_index = (N_tracer * i) + j
#            np.savetxt(os.path.join(current_dir, "sweep{}/forceout{}.dat".format(h+1, force_index)), 
#                np.column_stack((all_forces[:, 0], all_forces[:, j+1])))
#
