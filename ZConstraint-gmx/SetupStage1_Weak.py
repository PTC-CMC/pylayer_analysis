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
parser.add_option('--z0', action = 'store', type = 'float', dest = 'z0', default = 1.0)
parser.add_option('--Nwin', action = 'store', type = 'int', dest = 'N_window', default = 40)
parser.add_option('--Ntracer', action = 'store', type = 'int', dest = 'N_tracer', default = 8)
parser.add_option('--gro', action = 'store', type = 'string', dest = 'grofile')
parser.add_option('--k', action = 'store', type = 'float', dest = 'pull_coord_k', default = '40')
parser.add_option('--top', action = 'store', type = 'string', dest = 'topfile')
parser.add_option('--sweep', action='store', type='int', dest='sweep', default=0)
parser.add_option('--auto', action='store_true', dest='auto', default=False)
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
#os.system('echo q | gmx make_ndx -f {}'.format(grofile))

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
thing = SystemSetup(z0=z0, dz=dz, N_window=N_window, N_tracer=N_tracer,auto_detect=options.auto, grofile=grofile)
thing.gather_tracer(grofile = grofile)
#tracer_list = thing.get_Tracers()
tracer_list = thing.tracer_list
z_list = thing.zlist[0]*np.ones(len(tracer_list))
os.system('mkdir -p sweep{}'.format(options.sweep))
thing.write_zlist("sweep{0}/z_windows.out".format(options.sweep))
thing.write_tracerlist(thing._tracer_list, tracerlog="sweep{0}/tracers.out".format(options.sweep))



for i in range(N_sims):
    print('Z_windows: {}'.format(z_list))
    directoryname = 'sweep{}/Sim{}'.format(str(options.sweep), str(i))
    os.system('mkdir -p {}'.format(directoryname)) 
    thing.write_pulling_mdp(pull_filename=(directoryname + '/' + 'Stage1_Weak'+str(i)+'.mdp'), 
            tracerlist=tracer_list, z_window_list=z_list, 
            grofile=grofile, pull_coord_rate=0, pull_coord_k=pull_coord_k)
    mdpfile = str('Stage1_Weak' + str(i) + '.mdp')
    filename = str('Stage1_Weak' + str(i))
    thing.write_grompp_file(directoryname=directoryname, filename=filename, 
            grofile=grofile, mdpfile=mdpfile, indexfile=indexfile, topfile=topfile)

    os.system('cat {} {} > {}'.format('index.ndx', str(directoryname) + '/' + str('Stage1_Weak' + str(i) + '.ndx'), 'FullIndex.ndx'))
    os.system('cp {} {}'.format(indexfile, directoryname))
    os.system('cp {} {}'.format(grofile, directoryname))
    os.system('cp {} {}'.format(topfile, directoryname))

    z_list += dz


