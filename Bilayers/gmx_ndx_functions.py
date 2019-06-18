import mdtraj
import numpy as np
from collections import OrderedDict

#####################
## Functions to write gmx ndx files
## and get the gmx index selections related to that  group
###################


def get_index_selections():
    index_selections = OrderedDict({'DSPC':'0'})
    index_selections.update({'oh12':'1'})
    index_selections.update({'oh16':'2'})
    index_selections.update({'oh24':'3'})
    index_selections.update({'ffa12':'4'})
    index_selections.update({'ffa16':'5'})
    index_selections.update({'ffa24':'6'})
    index_selections.update({'non-water':'7'})
    index_selections.update({'HOH':'8'})
    index_selections.update({'oh18':'9'})
    index_selections.update({'oh20':'10'})
    index_selections.update({'oh22':'11'})
    index_selections.update({'ISIS':'12'})
    index_selections.update({'non-DSPC':'13'})


    return index_selections

def write_ndx(grofile='npt.gro', ndxfile='hbond.ndx', remove_midplane_atoms=True):
    traj = mdtraj.load(grofile)
    index_selections = get_index_selections()
    with open(ndxfile,'w') as f:
        for resname in index_selections.keys():
            if 'non-water' in resname:
                atom_indices = np.asarray(traj.topology.select('not water'))
            elif 'non-DSPC' in resname:
                atom_indices = np.asarray(traj.topology.select('not water and not resname DSPC'))
            else:
                atom_indices = np.asarray(traj.topology.select('resname {}'.format(resname)))
            if remove_midplane_atoms:
                midplane = np.mean(traj.unitcell_lengths[:,2])
            atom_indices = np.asarray(atom_indices)
            n_rows = atom_indices.shape[0] // 15
            remainder = atom_indices.shape[0] % 15
            f.write('[ {} ]\n'.format(resname))
            for i in range(n_rows):
                for j in range(15):
                    f.write('{}  '.format(atom_indices[15*i +j]+1))
                f.write('\n')
            for i in range(remainder,0,-1):
                f.write('{} '.format(atom_indices[-i]+1))
            f.write('\n')

if __name__ == "__main__":
    write_ndx()
