import os
import pdb
import sys
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import mdtraj as mdt
import MDAnalysis as mda
import MDAnalysis.analysis.hbonds as hbonds
import pandas as pd
import subprocess


""" Compute hydrogen bonds for a Z-constrained MD trajectory



"""
# In the simulation folder
curr_dir = os.getcwd()
directory_prefix = 'sweep'
sweep_dirs = [f for f in os.listdir(os.getcwd()) if directory_prefix in f and os.path.isdir(f)]
all_hbond_profiles = []
# In the sweep directory
for sweep_folder in sweep_dirs:
    print("Sweep: {0}".format(sweep_folder))
    os.chdir(os.path.join(os.getcwd(), sweep_folder))
    sweepdir = os.getcwd()

    tracers = np.loadtxt('tracers.out',dtype='string')
    n_tracer = np.shape(tracers)[0]

    z_windows = np.loadtxt('z_windows.out')
    n_windows = np.shape(z_windows)[0]

    sim_folders = [sim for sim in os.listdir(os.getcwd()) if 'Sim' in sim and os.path.isdir(sim)]
    n_sims = len(sim_folders)
    #hbond_profile = (n_tracer * n_sims) * [[0]]
    hbond_profile = np.zeros(n_tracer * n_sims)
    all_indices = []

    # Need to identify the tracer
    # In the Sim folder
    for sim_index, sim in enumerate(sim_folders):
        print("SIM: {0}".format(sim))
        os.chdir(os.path.join(sweepdir, sim))
        #pdbfile = [f for f in os.listdir(os.getcwd()) if '.pdb' in f and 'Stage5' in f]
        grofile = [f for f in os.listdir(os.getcwd()) if '.gro' in f and "#" not in f  and 'Stage5' in f][0]
        tprfile = [f for f in os.listdir(os.getcwd()) if '.tpr' in f and "#" not in f and 'Stage5' in f][0]
        pdbfile = grofile[:-4]+'.pdb'

        # If no pdbfile, make one
        if pdbfile not in os.listdir(os.getcwd()):
            with open('gmx_pdb.log', 'w') as outfile:
                p = subprocess.Popen('echo 0| gmx trjconv -f {0} -s {1} -o {2} -conect -pbc mol'.format(grofile, tprfile, pdbfile), shell=True, stdout=outfile, stderr=outfile)
                p.wait()

        # Compute hbonds for every tracer and output to a different file
        for tracer_index, tracer in enumerate(tracers):
            u = mda.Universe(pdbfile)
            hbond_analysis = hbonds.HydrogenBondAnalysis(u, 'resid {}'.format(tracer), 'resname HOH or resname SOL', distance=2.5, angle=150)
            hbond_analysis.run()
            hbond_analysis.generate_table()
            hbond_info = pd.DataFrame.from_records(hbond_analysis.table)
            hbond_dict={}
            for donor, acceptor in zip(hbond_info['donor_resnm'], hbond_info['acceptor_resnm']):
                key = donor + '-' + acceptor
                if key in hbond_dict:
                    hbond_dict[key] += 1
                else:
                    hbond_dict[key] = 1

            file_index = (tracer_index * n_sims) + sim_index
            all_indices.append(file_index)
            with open(os.path.join(sweepdir,'hbonds_mda_{0}.dat'.format(file_index)), 'w') as f:
                for key,value in hbond_dict.iteritems():
                   f.write('{:<20s}: {} \n'. format(key, value))
                   #hbond_profile[file_index].append(value)
                   hbond_profile[file_index] = value
    np.savetxt(os.path.join(sweepdir, 'hbond_profile_mda.dat'), hbond_profile)
    all_hbond_profiles.append(hbond_profile)
#    with open(os.path.join(sweepdir,'hbond_profile.dat'), 'w') as f:
#        print(sorted(all_indices))
#        for line in hbond_profile:
#            f.write("{4.4f}{4.4f}".format(np.mean(line), np.std(line)))
#            f.write("\n")
    os.chdir(curr_dir)

average_hbond_profile = np.mean(all_hbond_profiles, axis=0)
std_hbond_profile = np.std(all_hbond_profiles, axis=0)
whole_hbond_profile = np.column_stack((average_hbond_profile, std_hbond_profile))
np.savetxt('hbond_profile.dat', whole_hbond_profile)
z_windows = np.loadtxt('z_windows.out')

fig, ax = plt.subplots(1,1)
path = os.getcwd()
simulation = path.split('/')[-1]
composition = path.split('/')[-2]
l, = ax.plot(z_windows, whole_hbond_profile[:,0], label=composition)
ax.fill_between(z_windows, whole_hbond_profile[:,0] - whole_hbond_profile[:,1],
        whole_hbond_profile[:,0] + whole_hbond_profile[:,1], color = l.get_color(), alpha=0.25)
ax.set_ylabel("H-bonds")
ax.set_xlabel("Z coordinate (nm)")
ax.legend()
plt.tight_layout()
plt.savefig('hbond_profile.jpg', transparent=True)
