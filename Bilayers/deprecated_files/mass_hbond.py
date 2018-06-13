import os
import numpy as np
import subprocess
import occupied_area
import bilayer_analysis_functions
import mdtraj
import time

# Navigate through all the composition and simulation folders

# Basic file structure:
# ~/Trajectories/composition_folder/simulation_folder/simulation_files_here
working_dir = os.getcwd()
print("There are {} CPUs for pooling".format(os.cpu_count()))

with open("unsafe_sims.txt",'r') as f:
    unsafe_sims = [line.strip() for line in f]

z_bins = np.arange(1, 9, 0.1)




# Looping through each composition folder
for line in open("systems_of_interest.txt", 'r'):
#for line in open("testing_toc.txt", 'r'):
    composition_folder = line.strip()
    os.chdir(os.path.join(working_dir, composition_folder))
    simulation_folders = [f for f in os.listdir() if os.path.isdir(f)]

    # Looping through each simulation folder
    for simulation in simulation_folders: 
        os.chdir(os.path.join(working_dir, composition_folder, simulation))
        print("{:<25s}".format(simulation))

        # If this is a valid simulation then we do stuff
        if simulation not in unsafe_sims:
            pdbfile = "md_" + simulation + ".pdb"
            xtcfile = "md_" + simulation + ".xtc"
            tprfile = "md_" + simulation + ".tpr"
            grofile = "md_" + simulation + ".gro"
            last20 = "last20.xtc"
            # Check if the simulation even completed
            if xtcfile in os.listdir():
                # Make a correctly centered pdb file
                with open("gmx_pdb.log", 'w') as outfile:
                    p = subprocess.Popen("echo 'q' | gmx make_ndx -f md_{}.gro".format(simulation), shell=True, stdout=outfile,stderr=outfile)
                    p.wait()
                    p = subprocess.Popen("echo ' [a_53]' > newindex.ndx", shell=True, stdout=outfile, stderr=outfile)
                    p.wait()
                    p = subprocess.Popen("echo ' 53 ' >> newindex.ndx", shell=True, stdout=outfile, stderr=outfile)
                    p.wait()
                    p = subprocess.Popen("cat index.ndx >> newindex.ndx", shell=True, stdout=outfile, stderr=outfile)
                    p.wait()

                    # First center atom 53, to get the whole bilayer continuously in the box
                    p= subprocess.Popen("echo 0 1|gmx trjconv -f {0} -s {1} -o {2} -conect -center -n newindex.ndx -pbc mol".format(
                    grofile, tprfile, pdbfile), shell=True, stdout=outfile, stderr=outfile)
                    p.wait()

                    # Then center thee whole bilayer to make sure the bilayer is centered
                    p= subprocess.Popen("echo 1 0|gmx trjconv -f {0} -s {1} -o {2} -conect -center -b 100000 -e 100000 -pbc mol".format(
                    pdbfile, tprfile, pdbfile), shell=True, stdout=outfile, stderr=outfile)
                    p.wait()

                # And then do the hydrogen bond calc
                traj_pdb = mdtraj.load(pdbfile)
                if last20 in os.listdir():
                    traj = mdtraj.load(last20, top=grofile)
                else: 
                    traj = mdtraj.load(xtcfile, top=grofile)
                topol = traj.topology
                lipid_dict, headgroup_dict = bilayer_analysis_functions.get_lipids(topol)
                (hbond_matrix_avg, hbond_matrix_std, hbond_matrix_list, labelmap)  = \
                    bilayer_analysis_functions.calc_hbonds(traj, traj_pdb, topol,
                        lipid_dict, headgroup_dict, include_water_solute=True)
                with open('hbonds.dat', 'w') as f:
                    for row_label in labelmap.keys():
                        for col_label in labelmap.keys():
                            row_index = labelmap[row_label]
                            col_index = labelmap[col_label]
                            hbond_avg = hbond_matrix_avg[row_index, col_index]
                            hbond_std = hbond_matrix_std[row_index, col_index]
                            f.write('{:<20s}: {} ({})\n'.format(str(row_label+"-"+ col_label), hbond_avg, hbond_std))


        else:
            print("Simulation not considered safe")

        print("------------------")


    # Go back to ~/Trajectories
    os.chdir(working_dir)
