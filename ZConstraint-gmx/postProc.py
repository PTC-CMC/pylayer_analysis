import os
import numpy as np
import sys
import subprocess
import pdb
"""
Each Stage 5 simulation has a _pullf.xvg file that has forces for each tracer
Split each _pullf.xvg file into individual files for each tracer, called forceout

Notes
-----
Need to be careful about file numbering, 
Within a Sim_pullf.xvg file, each file is forceout_i, forceout_i+1*N_sim, forceout_i+2*N_sim, etc
Example: 40 windows, 8 tracers, 5 simulations
Tracers in sim0 correspond to like forceout0,5,10,15,20,25,30,35
Tracers in sim1 correspond to like forceout1,6,11,16,21,26,31,36
"""

# This could be done reading tracers.out and z_windows.out
N_sim = 5
N_trace = 8
process = subprocess.Popen("cp z_windows.out y0list.txt".split(), stdout=subprocess.PIPE)
output, error = process.communicate()
filenames = ["{}/Sim0/Stage5_ZCon0_pullf.xvg".format(os.getcwd()),
        "{}/Sim1/Stage5_ZCon1_pullf.xvg".format(os.getcwd()),
        "{}/Sim2/Stage5_ZCon2_pullf.xvg".format(os.getcwd()),
        "{}/Sim3/Stage5_ZCon3_pullf.xvg".format(os.getcwd()),
        "{}/Sim4/Stage5_ZCon4_pullf.xvg".format(os.getcwd())]

# Read in the pullf xvg file
for i, Sim in enumerate(filenames):
    with open(Sim) as f:
        rawlines = list(f)
    
    # Filter out the useless lines in the xvg file
    all_data=[]
    for q, entryline in enumerate(rawlines):
        if '#' not in entryline and '@' not in entryline: 
            items = entryline.split()
            all_data.append(items)
    
    sim_i = 0 
    # Generate a forceout file for each tracer
    for k in range(N_trace):
        index = i + k*N_sim
        #pdb.set_trace()
        forcefilename = 'forceout{}'.format(index)
        forcefile = open(forcefilename,'w')
        for timestep in range(len(all_data)):
            forcefile.write('{}\t{}\n'.format(all_data[timestep][0], abs(float(all_data[timestep][k+1]))))
        forcefile.close()
