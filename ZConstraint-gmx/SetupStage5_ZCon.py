from SystemSetup import *
import numpy as np
from optparse import OptionParser
import os
'''
SetupStage5_ZCon:
    Z-Constrained molecular dynamics, recording restraining forces
    Tracers should be at different z-windows and well-equilibrated
    Define another pulling potential around that z-window
    Run time is same across all simulation
    Do this N_window/N_tracer times for each z-window
    Will need to re-grompp with new top, mdp, gro files

'''


#Read in tracers
parser = OptionParser()
parser.add_option('--t', action = 'store', type = 'string', dest = 'tracerfile', default = 'tracers.out')
parser.add_option('--z', action = 'store', type = 'string', dest = 'zwindows', default  = 'z_windows.out')
parser.add_option('--top', action = 'store', type = 'string', dest = 'topfile', default = 'RedonepureDSPC.top')
parser.add_option('--k', action = 'store', type = 'float', dest = 'pull_coord_k', default = '1000')
parser.add_option('--sweep', action='store', type='int', dest='sweep', default=0)
(options, args) = parser.parse_args()

thing = SystemSetup(z_windows=options.zwindows)
tracerlist_filename = options.tracerfile
zwindows_filename = options.zwindows
topfile = options.topfile
pull_coord_k = options.pull_coord_k
pull_coord_rate = 0
indexfile = 'FullIndex.ndx'
duration = 2e3 # Run z-constraining and record forces for 2 ns

#Read tracers
#tracerlist = open(tracerlist_filename, 'r')
#tracerlistlines = tracerlist.readlines()
thing.read_tracers(tracerlist_filename)
#N_tracer = len(tracerlistlines)
N_tracer = len(thing.tracer_list)


#Read zwindows
#zwindows = open(zwindows_filename, 'r')
#zwindowslines = zwindows.readlines()
thing.read_zlist(zwindows_filename)
#N_window = len(zwindowslines)
N_window = len(thing.zlist)

N_sims = int(N_window / N_tracer)
#dz = np.round(float(thing.get_dz()), 3)
#z0 = np.round(float(thing.get_z0()), 3)
dz = thing.dz
z0 = thing.z0

print('Setup Stage 5 ZConstraining: Z-Constrained MD')
print('{:10s} = {}'.format('dz', dz))
print('{:10s} = {}'.format('z0', z0))
print('{:10s} = {}'.format('N_window', N_window))
print('{:10s} = {}'.format('N_tracer', N_tracer))
print('{:10s} = {}'.format('Tracerfile', tracerlist_filename))
print('{:10s} = {}'.format('Zwindows', zwindows_filename))


#tracer_list = thing.get_Tracers()
tracer_list = thing.tracer_list
z_list = z0*np.ones(len(tracer_list))
#With references, each tracer will be moving to a different location, unlike stages 1 and 2
#Need to modify z_list to account for each tracer hitting different zwindows now
z_sep = dz*N_window/N_tracer
for i in range(len(z_list)):
    z_list[i] = z_list[i] + (z_sep*i)

#Need to write position restraint file for mdp file

for i in range(N_sims):
    print('Z_windows: {}'.format(z_list))
    directoryname = 'sweep{}/Sim{}'.format(str(options.sweep), str(i))
    mdpfile = str('Stage5_ZCon' + str(i) + '.mdp')
    filename = str('Stage5_ZCon' + str(i))
    oldfilename = str('Stage4_Eq' + str(i))
    cptfile = (oldfilename + '.cpt')
    oldtpr = (oldfilename + '.tpr')
    grofile = (directoryname + '/' + oldfilename+'.gro')
    oldgrofile = (oldfilename + '.gro')
    curr_dir_grofile = (oldfilename+'.gro')
    thing.write_pulling_mdp(directoryname + '/' + 'Stage5_ZCon'+str(i)+'.mdp', tracer_list, z_list, grofile, pull_coord_rate =
            pull_coord_rate, pull_coord_k = pull_coord_k, t_pulling = duration, stagefive = True)

    thing.write_grompp_file(directoryname, filename, oldgrofile, mdpfile, indexfile, oldtpr=oldtpr, cptfile=cptfile, topfile=topfile)

    z_list += dz

