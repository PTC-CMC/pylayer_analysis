from SystemSetup import *
import numpy as np
from optparse import OptionParser
import os
import subprocess
'''
SetupStage2_Strong:
    Continues the respective weak pulling simulations, but with strong pulling constants
    Pulling to the same z-windows as the weak pulling simulatino
    Do this N_window/N_tracer times for different z-windows
    Writes mdp and job submission files
'''

#Plan: use grompp with a new mdp, send in old tpr, write new tpr, then mdrun with new tpr 

#Read in tracers
parser = OptionParser()
parser.add_option('--t', action = 'store', type = 'string', dest = 'tracerfile', default = 'tracers.out')
parser.add_option('--z', action = 'store', type = 'string', dest = 'zwindows', default  = 'z_windows.out')
parser.add_option('--top', action = 'store', type = 'string', dest = 'topfile', default = 'RedonepureDSPC.top')
parser.add_option('--k', action = 'store', type = 'float', dest = 'pull_coord_k', default = '500')
parser.add_option('--sweep', action='store', type='int', dest='sweep', default=0)
(options, args) = parser.parse_args()

thing = SystemSetup(z_windows=options.zwindows)
tracerlist_filename = options.tracerfile
zwindows_filename = options.zwindows
pull_coord_k = options.pull_coord_k
topfile = options.topfile
indexfile = 'FullIndex.ndx'
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
#dz = np.round(float(thing.get_dz()), 3)
dz = thing.dz
#z0 = np.round(float(thing.get_z0()), 3)
z0 = thing.z0

print('Setup Stage 2 Strong: Pulling with strong, fixed reference')
print('{:10s} = {}'.format('dz', dz))
print('{:10s} = {}'.format('z0', z0))
print('{:10s} = {}'.format('N_window', N_window))
print('{:10s} = {}'.format('N_tracer', N_tracer))
print('{:10s} = {}'.format('Tracerfile', tracerlist_filename))
print('{:10s} = {}'.format('Zwindows', zwindows_filename))
print('{:10s} = {}'.format('k', pull_coord_k))

#tracer_list = thing.get_Tracers()
tracer_list = thing.tracer_list
z_list = thing.zlist[0]*np.ones(len(tracer_list))
#p = subprocess.Popen("cp {} sweep{}".format(zwindows_filename, options.sweep))
#p.wait()

for i in range(N_sims):
    #print('Writing mdp and submit files for k = {} pulling to z = {}'.format(pull_coord_k, np.round(z_list[0],2)))
    print('Z_windows: {}'.format(z_list))
    directoryname = 'sweep{}/Sim{}'.format(str(options.sweep), str(i))
    mdpfile = str('Stage2_Strong' + str(i) + '.mdp')
    filename = str('Stage2_Strong' + str(i))
    oldfilename = str('Stage1_Weak' + str(i))
    cptfile = (oldfilename + '.cpt')
    oldtpr = (oldfilename + '.tpr')
    grofile = (directoryname+'/'+oldfilename+'.gro')
    oldgrofile = (oldfilename + '.gro')
    thing.write_pulling_mdp(directoryname + '/' + 'Stage2_Strong'+str(i)+'.mdp', tracer_list, z_list, grofile, pull_coord_rate = 0,
            pull_coord_k = pull_coord_k)
    thing.write_grompp_file(directoryname, filename, oldgrofile, mdpfile, indexfile, oldtpr=oldtpr, cptfile=cptfile, topfile = topfile) 

    z_list += dz

