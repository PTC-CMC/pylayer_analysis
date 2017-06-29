# Run this to write the pbs script for rahman
# This pbs script executs the 5 stages of the permeability simulations

from optparse import OptionParser
import os
parser = OptionParser()
parser.add_option('-n', action = 'store', type = 'int', dest = 'n_sweeps', default = '1')
(options, args) = parser.parse_args()

curr_dir = os.getcwd()
simulation = curr_dir.split('/')[-1]
composition = curr_dir.split('/')[-2]
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
    f.write("cd ~/Trajectories/Data/{}/{}\n".format(composition,simulation))
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
            f.write("\t python Setup{0}.py --gro md_{1}.gro --top {1}.top --sweep $i \n".format(prefix, simulation))
        else:
            f.write("\t python Setup{0}.py --top {1}.top --sweep $i --t sweep$i/tracers.out --z sweep$i/z_windows.out\n".format(prefix,simulation))

        # Write the grompp line 
        f.write("\t python massRegrompp.py --Stage {} --sweep $i\n".format(stagenumber))

        # Write all the mdrun lines
        for simnumber in range(5):
            f.write("\t gmx mdrun -ntomp 8 -gpu_id 0 -deffnm ~/Trajectories/Data/{0}/{1}/sweep$i/Sim{2}/{3}{2} > ~/Trajectories/Data/{0}/{1}/sweep$i/{3}{2}.out 2>&1\n".format(composition, simulation, simnumber, prefix))

        f.write("\t \n")
    f.write("done\n")
print("Written to {}_permeability.pbs".format(simulation))
