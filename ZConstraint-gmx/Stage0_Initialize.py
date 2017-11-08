# Run this to write the pbs script for rahman
# This pbs script executs the 5 stages of the permeability simulations

from optparse import OptionParser
import mdtraj
import os
parser = OptionParser()
parser.add_option('-n', action='store', type='int', dest='n_sweeps', default='1')
parser.add_option('--dz', action='store', type='float', dest='dz', default=0.2)
parser.add_option('--z0', action='store', type='float', dest='z0', default= 1.0)
parser.add_option('--gro', action = 'store', type = 'string', dest = 'grofile')
parser.add_option('--Nwin', action='store', type='int', dest='N_window', default=40)
parser.add_option('--Ntracer', action='store', type='int', dest='N_tracer', default=8)
parser.add_option('--auto', action='store_true', dest='auto', default=False)
parser.add_option('--nocenter', action='store_true', dest='nocenter', default=False)
(options, args) = parser.parse_args()


curr_dir = os.getcwd()
simulation = curr_dir.split('/')[-1]
composition = curr_dir.split('/')[-2]
N_sims = int(options.N_window/options.N_tracer)
initial_configuration = 'md_{}.gro'.format(simulation)
# Center gro file
if not options.nocenter:
    traj = mdtraj.load(options.grofile)
    perma_traj = mdtraj.load(options.grofile)
    non_water = traj.topology.select('not water')
    sub_traj = traj.atom_slice(non_water)
    # Get center of mass of the bilayer 
    com = mdtraj.compute_center_of_mass(sub_traj)
    box = [traj.unitcell_lengths[0,0], traj.unitcell_lengths[0,1], traj.unitcell_lengths[0,2]]

    # Shift all coordinates so bilayer is at center of box
    for i, val in enumerate(traj.xyz):
        traj.xyz[i] = [[x, y, z-com[0,2]+(box[2]/2)] for x,y,z in traj.xyz[i][:]]

    # Go back and fix for pbc
    for i, val in enumerate(traj.xyz):
        for j,atom in enumerate(traj.xyz[i]):
            for k, coord in enumerate(traj.xyz[i,j]):
                if coord < 0:
                    traj.xyz[i,j,k] += box[k]
                elif coord > box[k]:
                    traj.xyz[i,j,k] -= box[k]
    traj.save('centered.gro')
    initial_configuration = 'centered.gro'


if options.auto:
    auto_flag = "--auto"
else:
    auto_flag = ""

with open("{}_permeability.pbs".format(simulation),'w') as f:

    # Simple pbs header
    f.write("#!/bin/sh -l\n")
    f.write("#PBS -N {}_perm\n".format(simulation))
    f.write("#PBS -l nodes=1:ppn=16\n")
    f.write("#PBS -l walltime=96:00:00\n")
    f.write("#PBS -q low\n")
    f.write("#PBS -m abe\n")
    f.write("#PBS -M alexander.h.yang@vanderbilt.edu\n")
    f.write("\n")
    f.write("\n")
    f.write("cd $PBS_O_WORKDIR\n")
    f.write("echo `cat $PBS_NODEFILE`\n")
    f.write("\n")
    f.write("module load gromacs/5.1.0\n")
    f.write("cd {}\n".format(curr_dir))
    f.write("for i in {{0..{}}}\n".format(options.n_sweeps-1))
    f.write("do\n")
    # Now the repetitive setting up, grompping, mdrunning for stages 1-5
    for stagenumber in range(1,6):
        # Determine the prefixes for subsequent string writing
        stages = {1: "Stage1_Weak", 2: "Stage2_Strong", 3: "Stage3_Moving", 
                4: "Stage4_Eq", 5: "Stage5_ZCon"}
        prefix = stages[stagenumber]

        # Write the setup lines
        if stagenumber == 1:
            f.write("\t python Setup{0}.py --gro {7} --top {1}.top --sweep $i --dz {2} --z0 {3} --Nwin {4} --Ntracer {5} {6}\n".format(prefix, simulation, 
                options.dz, options.z0, options.N_window, options.N_tracer, 
                auto_flag, initial_configuration))
        else:
            f.write("\t python Setup{0}.py --top {1}.top --sweep $i --t sweep$i/tracers.out --z sweep$i/z_windows.out\n".format(prefix,simulation))

        # Write the grompp line 
        f.write("\t python massRegrompp.py --Stage {} --sweep $i\n".format(stagenumber))

        # Write all the mdrun lines
        for simnumber in range(N_sims):
            #f.write("\t gmx mdrun -ntomp 8 -gpu_id 0 -deffnm ~/Trajectories/Data/{0}/{1}/sweep$i/Sim{2}/{3}{2} > ~/Trajectories/Data/{0}/{1}/sweep$i/{3}{2}.out 2>&1\n".format(composition, simulation, simnumber, prefix))
            f.write("\t gmx mdrun -deffnm {0}/sweep$i/Sim{1}/{2}{1} > {0}/sweep$i/{2}{1}.out 2>&1\n".format(curr_dir,simnumber, prefix))
            f.write("rm \"#index\"* \n")

        f.write("\t \n")
    f.write("done\n")
print("Written to {}_permeability.pbs".format(simulation))
