import os
import time
import numpy as np
import mdtraj 
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-f', action="store", type="string", default = 'nopbc.xtc', dest = 'trajectory')
parser.add_option('-p', action="store", type="string", default = 'nopbc.pdb', dest = 'pdb')
parser.add_option('-o', action="store", type="string", default = 'hbonds.txt', dest = 'outfile')
(options, args) = parser.parse_args()

traj_pdb = mdtraj.load(options.trajectory, top=options.pdb)
topol = traj_pdb.topology
outfile = open(options.outfile, 'w')


start = time.time()
hbonds = mdtraj.wernet_nilsson(traj_pdb[0:-1:5], exclude_water = True, include_water_solute = True)
end = time.time()
duration = end - start
outfile.write('{:<20s}: {}\n'.format('Trajectory',options.trajectory))
outfile.write('{:<20s}: {}\n'.format('Structure',options.pdb))

total_hbonds=[]
# hbonds is just a list of lists, the outer list corresponds to each fframe
# the inner list is a list of each hbonding triplet
for hbond_frame in hbonds:
    total_hbonds.append(np.shape(hbond_frame)[0])
with_avg = np.mean(total_hbonds)
with_std = np.std(total_hbonds)

outfile.write("{:<20}{}({})\t{:>10s}{:>10}\n".format("yes water-solute:", with_avg,with_std, 'Duration:',duration))

start = time.time()
hbonds = mdtraj.wernet_nilsson(traj_pdb[0:-1:5], exclude_water = True, include_water_solute = False)
end = time.time()
duration = end - start
total_hbonds=[]
# hbonds is just a list of lists, the outer list corresponds to each fframe
# the inner list is a list of each hbonding triplet
for hbond_frame in hbonds:
    total_hbonds.append(np.shape(hbond_frame)[0])
without_avg = np.mean(total_hbonds)
without_std = np.std(total_hbonds)


outfile.write("{:<20}{}({})\t{:>10s}{:>10}\n".format("no water-solute:", without_avg,without_std, 'Duration:', duration))
outfile.write("************\n")

outfile.close()
