import sys
import subprocess
import mdtraj
import group_templates


##############
## Compute Scd (carbon-deuterium) order parameters
## Look at each residue
## For each lipid tail
## Looks at tails 1 and 2 for 2-tailed molecules
## And then looks at the tails of the single-tailed molecules
#############

def main(grofile='npt.gro', xtcfile='npt_80-100ns.xtc', tprfile='npt.tpr'):
    traj = mdtraj.load(grofile)
    c36_groups = group_templates.charmm36_groups()
    # One tail per ndx
    # Need to have a group for all C36 atoms, another for all C37 atoms etc
    unique_resnames = set([r.name for r in traj.topology.residues if not r.is_water])

    tails = {}
    for resname in unique_resnames:
        if ('tail_1' in c36_groups[resname].keys() and 
                'tail_2' in c36_groups[resname].keys()):
            tails[resname+'a'] = []
            tails[resname+'b'] = []
            # Find the first instance of this residue just to get some atom names
            single_residue = [r for r in traj.topology.residues 
                    if r.name == resname][0]
            for base_index in c36_groups[resname]['tail_1']:
                if 'C' == single_residue.atom(base_index).name[0]:
                    tails[resname+'a'].append(single_residue.atom(base_index).name)
            for base_index in c36_groups[resname]['tail_2']:
                if 'C' == single_residue.atom(base_index).name[0]:
                    tails[resname+'b'].append(single_residue.atom(base_index).name)

            with open('{}.ndx'.format(resname+'a'), 'w') as f:
                for atomname in tails[resname+'a']:
                    f.write('[  {}  ]\n'.format(atomname))
                    atom_indices = traj.topology.select(
                            'resname {} and name {}'.format(resname, atomname))
                    n_rows = atom_indices.shape[0] // 15
                    remainder = atom_indices.shape[0] % 15
                    for i in range(n_rows):
                        for j in range(15):
                            f.write('{}  '.format(atom_indices[15*i +j]+1))
                        f.write('\n')
                    for i in range(remainder,0,-1):
                        f.write('{} '.format(atom_indices[-i]+1))
                    f.write('\n')
            with open('{}.ndx'.format(resname+'b'), 'w') as f:
                for atomname in tails[resname+'b']:
                    f.write('[  {}  ]\n'.format(atomname))
                    atom_indices = traj.topology.select(
                            'resname {} and name {}'.format(resname, atomname))
                    n_rows = atom_indices.shape[0] // 15
                    remainder = atom_indices.shape[0] % 15
                    for i in range(n_rows):
                        for j in range(15):
                            f.write('{}  '.format(atom_indices[15*i +j]+1))
                        f.write('\n')
                    for i in range(remainder,0,-1):
                        f.write('{} '.format(atom_indices[-i]+1))
                    f.write('\n')


        elif 'tail' in c36_groups[resname].keys():
            # Find the first instance of this residue just to get some atom names
            tails[resname] = []
            single_residue = [r for r in traj.topology.residues 
                    if r.name == resname][0]
            for base_index in c36_groups[resname]['tail']:
                if 'C' == single_residue.atom(base_index).name[0]:
                    tails[resname].append(
                            single_residue.atom(base_index).name)
            with open('{}.ndx'.format(resname), 'w') as f:
                for atomname in tails[resname]:
                    f.write('[  {}  ]\n'.format(atomname))
                    atom_indices = traj.topology.select(
                            'resname {} and name {}'.format(resname, atomname))
                    n_rows = atom_indices.shape[0] // 15
                    remainder = atom_indices.shape[0] % 15
                    for i in range(n_rows):
                        for j in range(15):
                            f.write('{}  '.format(atom_indices[15*i +j]+1))
                        f.write('\n')
                    for i in range(remainder,0,-1):
                        f.write('{} '.format(atom_indices[-i]+1))
                    f.write('\n')


    for prefix in tails.keys():
        p = subprocess.Popen(
                'gmx order -f {0} -s {1} -n {2}.ndx -od {2}_scd.xvg'.format(
                    xtcfile, tprfile, prefix), shell=True, stdout=sys.stdout, 
                stderr=sys.stderr)
        p.wait()

if __name__ == "__main__":
    main()
