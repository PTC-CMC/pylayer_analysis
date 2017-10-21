import os
import subprocess

""" After running SetupStage5_ZCon_lmps.py
This script just submits all the job scripts to the cluster"""

working_dir = os.getcwd()
sweep_folders = [f for f in os.listdir() if os.path.isdir(f) and 'sweep' in f]
for sweep in sweep_folders:
    os.chdir(os.path.join(working_dir, sweep))
    z_windows_file = os.path.join(working_dir, sweep, 'z_windows.out')
    tracerfile = os.path.join(working_dir, sweep, 'tracers.out')
    sim_folders = [g for g in os.listdir() if os.path.isdir(g) and 'Sim' in g]
    for sim in sim_folders:
        os.chdir(os.path.join(working_dir, sweep, sim))
        p = subprocess.Popen('sbatch Stage5_ZCon_lmps.sbatch', shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

