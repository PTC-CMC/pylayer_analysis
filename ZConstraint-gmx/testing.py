from SystemSetup import *

import numpy as np
from optparse import OptionParser
import os
import pdb
import mdtraj
import mbuild as mb

grofile = "RedonepureDSPC.gro"
traj = mdtraj.load(grofile)
perma_traj = mdtraj.load(grofile)
non_water = traj.topology.select('not water')
sub_traj = traj.atom_slice(non_water)
# Get center of mass of the bilayer 
com = mdtraj.compute_center_of_mass(sub_traj)
box = [traj.unitcell_lengths[0,0], traj.unitcell_lengths[0,1], traj.unitcell_lengths[0,2]]

# Shift all coordinates so bilayer is at center of box
for i, val in enumerate(traj.xyz):
    traj.xyz[i] = [[x, y, z-com[0,2]+(box[2]/2)] for x,y,z in traj.xyz[i][:]]
traj.save('pbc.gro')

# Go back and fix for pbc
for i, val in enumerate(traj.xyz):
    for j,atom in enumerate(traj.xyz[i]):
        for k, coord in enumerate(traj.xyz[i,j]):
            if coord < 0:
                traj.xyz[i,j,k] += box[k]
            elif coord > box[k]:
                traj.xyz[i,j,k]-= box[k]
            #if x < 0:
            #    x += box[0]
            #if x > box[0]:
            #    x -= box[0]
            #if y < 0:
            #    y += box[1]
            #if y > box[1]:
            #    y -= box[1]
            #if z < 0:
            #    z += box[2]
            #if z > box[2]:
            #    z -= box[2]
pdb.set_trace()
traj.save('nopbc.gro')
#thing = SystemSetup(grofile=grofile, auto_detect=True)
