from SystemSetup import *
import numpy as np
from optparse import OptionParser
import os
import subprocess

#Read in tracers
parser = OptionParser()
parser.add_option('--t', action = 'store', type = 'string', dest = 'tracerfile', default = 'tracers.out')
parser.add_option('--z', action = 'store', type = 'string', dest = 'zwindows', default  = 'z_windows.out')
(options, args) = parser.parse_args()

thing = SystemSetup()
tracerlist_filename = options.tracerfile
zwindows_filename = options.zwindows

#Read tracers
tracerlist = open(tracerlist_filename, 'r')
tracerlistlines = tracerlist.readlines()
thing.read_tracers(tracerlistlines)
N_tracer = len(tracerlistlines)


#Read zwindows
zwindows = open(zwindows_filename, 'r')
zwindowslines = zwindows.readlines()
thing.read_zlist(zwindowslines)
N_window = len(zwindowslines)

N_sims = int(N_window / N_tracer)

# Read in the forces files, splitting 
# Them into different force files
current_dir = os.getcwd()
for i in range(N_sims):
    filename = "Stage5_ZCon"+str(i)+"_pullf.xvg"
    os.chdir(os.path.join(current_dir, "Sim{}".format(i)))
    all_forces = np.loadtxt(filename, skiprows=30)
    for j in range(N_tracer):
        force_index = (N_tracer * i) + j
        np.savetxt(os.path.join(current_dir, "Forces/forceout{}.dat".format(force_index)), 
                np.column_stack((all_forces[:, 0], all_forces[:, j+1])))
