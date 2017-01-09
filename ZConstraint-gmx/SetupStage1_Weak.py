from SystemSetup import *
import numpy as np
from optparse import OptionParser
import os
'''
SetupStage1_Weak:
    Sets up pulling simulations with weak, fixed references to the z-window. 
    One pulling simulation will pull each tracer to the particular z-window
    Do this N_window/N_tracer times for different z-windows
    Simulations start from the same equilibrated structure
    Writes mdp, ndx, and the job submission files 
'''
parser = OptionParser()
parser.add_option('--dz', action = 'store', type = 'float', dest = 'dz', default = 0.2)
parser.add_option('--z0', action = 'store', type = 'float', dest = 'z0', default = 0.0)
parser.add_option('--Nwin', action = 'store', type = 'int', dest = 'N_window', default = 40)
parser.add_option('--Ntracer', action = 'store', type = 'int', dest = 'N_tracer', default = 8)
parser.add_option('--gro', action = 'store', type = 'string', dest = 'grofile', default = 'RedonepureDSPC.gro')
parser.add_option('--k', action = 'store', type = 'float', dest = 'pull_coord_k', default = '40')
parser.add_option('--top', action = 'store', type = 'string', dest = 'topfile', default =
    'RedonepureDSPC.top')
(options, args) = parser.parse_args()

dz = options.dz
z0 = options.z0
N_window = options.N_window
N_tracer = options.N_tracer
grofile = options.grofile
topfile = options.topfile
pull_coord_k = options.pull_coord_k
N_sims = int(N_window/N_tracer)
indexfile = 'FullIndex.ndx'
os.system('echo q | gmx make_ndx -f {}'.format(grofile))

print('Setup Stage 1 Weak: Pulling with weak, fixed reference')
print('{:10s} = {}'.format('dz', dz))
print('{:10s} = {}'.format('z0', z0))
print('{:10s} = {}'.format('N_window', N_window))
print('{:10s} = {}'.format('N_tracer', N_tracer))
print('{:10s} = {}'.format('Grofile', grofile))
print('{:10s} = {}'.format('Topfile', topfile))
print('{:10s} = {}'.format('k', pull_coord_k))

#For N_windows and N_tracers, need to set up N_windows/N_tracer simulations that pull 
# each tracer to the same z_window at dz intervals
thing = SystemSetup(z0 = z0, dz = dz, N_window = N_window, N_tracer = N_tracer)
thing.gather_tracer(grofile = grofile)
#tracer_list = thing.get_Tracers()
tracer_list = thing.tracer_list
z_list = z0*np.ones(len(tracer_list))



for i in range(N_sims):
    #print('Writing mdp, ndx, and submit files for k = {} pulling to z = {}'.format(pull_coord_k, np.round(z_list[0],2)))
    print('Z_windows: {}'.format(z_list))
    directoryname = 'Sim{}'.format(str(i))
    os.system('mkdir -p {}'.format(directoryname)) 
    thing.write_pulling_mdp(directoryname + '/' + 'Stage1_Weak'+str(i)+'.mdp', tracer_list, z_list, grofile, pull_coord_rate = 0, pull_coord_k = pull_coord_k)
    mdpfile = str('Stage1_Weak' + str(i) + '.mdp')
    filename = str('Stage1_Weak' + str(i))
    #Need to double up and fit two grompps + mdruns into one pbs file
    #thing.write_pbs(directoryname, filename, mdpfile, grofile, topfile)
    #Individual slurm files though
    #thing.write_slurm(directoryname, filename, mdpfile, grofile, topfile)
    thing.write_grompp_file(directoryname, filename, grofile, mdpfile, indexfile, topfile = topfile)

    os.system('cat {} {} > {}'.format('index.ndx', str(directoryname) + '/' + str('Stage1_Weak' + str(i) + '.ndx'), 'FullIndex.ndx'))
    os.system('cp {} {}'.format(indexfile, directoryname))
    os.system('cp {} {}'.format(grofile, directoryname))
    os.system('cp {} {}'.format(topfile, directoryname))

    z_list += dz

#for i in range(0, N_sims, 2):
#    directoryname1 = 'Sim{}'.format(str(i))
#    directoryname2 = 'Sim{}'.format(str(i+1))
#
#
#    mdpfile1 = str(directoryname1 + '/' + 'Stage1_Weak' + str(i) + '.mdp')
#    mdpfile2 = str(directoryname2 + '/' + 'Stage1_Weak' + str(i+1) + '.mdp')
#    pbsname = 'Stage1_Weak{}and{}'.format(str(i), str(i+1))
#    if i == N_sims - 1:
#        pbsname = 'Stage1_Weak{}'.format(str(i))
#        thing.write_pbs_single(pbsname, grofile, topfile, directoryname1, mdpfile1)
#    else:
#        thing.write_pbs(pbsname, grofile, topfile, directoryname1, mdpfile1, directoryname2, mdpfile2)
#
#
#
