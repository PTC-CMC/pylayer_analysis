import sys
import subprocess
import mdtraj
import group_templates


##############
## Compute Scd (carbon-deuterium) order parameters
## For each lipid tail
## Looks at tails 1 and 2 for 2-tailed molecules
## And then looks at the tails of the single-tailed molecules
#############

def main(grofile='npt.gro', xtcfile='npt_80-100ns.xtc', tprfile='npt.tpr'):
    c36_groups = group_templates.charmm36_groups()
    # One tail per ndx
    # Need to have a group for all C36 atoms, another for all C37 atoms etc
    unique_resnames = set([r.name for r in traj.topology.residues if not r.is_water])

    tail1_atomnames = []
    tail2_atomnames = []
    tail_other_atomnames = []
    for resname in unique_resnames:
        if 'tail_1' in c36_groups[resname].keys():
            # Find the first instance of this residue just to get some atom names
            single_residue = [r for r in traj.topology.residues 
                    if r.name == resname][0]
            for base_index in c36_groups[resname]['tail_1']:
                tail1_atomnames.append(single_residue.atom(base_index).name)
        # elif 'tail_2' in c36_groups[resname].keys():
            # Find the first instance of this residue just to get some atom names
            #single_residue = [r for r in traj.topology.residues 
                    #if r.name == resname][0]
            for base_index in c36_groups[resname]['tail_2']:
                tail2_atomnames.append(single_residue.atom(base_index).name)
        elif 'tail' in c36_groups[resname].keys():
            # Find the first instance of this residue just to get some atom names
            single_residue = [r for r in traj.topology.residues 
                    if r.name == resname][0]
            for base_index in c36_groups[resname]['tail']:
                tail_other_atomnames.append(single_residue.atom(base_index).name)

    with open('tail1.ndx', 'w') as f:
        for atomname in tail1_atomnames:
            f.write('[  {}  ]\n'.format(atomname))
            atom_indices = traj.topology.select('name {}'.format(atomname))
            n_rows = atom_indices.shape[0] // 15
            remainder = atom_indices.shape[0] % 15
            for i in range(n_rows):
                for j in range(15):
                    f.write('{}  '.format(atom_indices[15*i +j]+1))
                f.write('\n')
            for i in range(remainder,0,-1):
                f.write('{} '.format(atom_indices[-i]+1))
            f.write('\n')

    with open('tail2.ndx', 'w') as f:
        for atomname in tail2_atomnames:
            f.write('[  {}  ]\n'.format(atomname))
            atom_indices = traj.topology.select('name {}'.format(atomname))
            n_rows = atom_indices.shape[0] // 15
            remainder = atom_indices.shape[0] % 15
            for i in range(n_rows):
                for j in range(15):
                    f.write('{}  '.format(atom_indices[15*i +j]+1))
                f.write('\n')
            for i in range(remainder,0,-1):
                f.write('{} '.format(atom_indices[-i]+1))
            f.write('\n')

    with open('tail_other.ndx', 'w') as f:
        for atomname in tail_other_atomnames:
            f.write('[  {}  ]\n'.format(atomname))
            atom_indices = traj.topology.select('name {}'.format(atomname))
            n_rows = atom_indices.shape[0] // 15
            remainder = atom_indices.shape[0] % 15
            for i in range(n_rows):
                for j in range(15):
                    f.write('{}  '.format(atom_indices[15*i +j]+1))
                f.write('\n')
            for i in range(remainder,0,-1):
                f.write('{} '.format(atom_indices[-i]+1))
            f.write('\n')

    for prefix in ['tail1', 'tail2', 'tail_other']:
        p = subprocess.Popen('gmx order -f {0} -s {1} -n {2}.ndx -od {2}_scd.xvg'.format(
            xtcfile, tprfile, prefix), shell=True, stdout=sys.stdout, 
            stderr=sys.stderr)
        p.wait()

if __name__ == "__main__":
    main()
