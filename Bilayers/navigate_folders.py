import os
import numpy as np
import subprocess
import time

# Navigate through all the composition and simulation folders

# Basic file structure:
# ~/Trajectories/composition_folder/simulation_folder/simulation_files_here
#table_of_contents = open("table_of_contents.txt",r)
working_dir = os.getcwd()
total = 0
count = 0
# Looping through each composition folder
for line in open("systems_of_interest.txt", 'r'):
#for line in open("testing_toc.txt", 'r'):
    composition_folder = line.strip()
    os.chdir(os.path.join(working_dir, composition_folder))
    simulation_folders = [f for f in os.listdir() if os.path.isdir(f)]

    # Looping through each simulation folder
    for simulation in simulation_folders:
        os.chdir(os.path.join(working_dir, composition_folder, simulation))
        if 'occupied_area.dat' in os.listdir():
            print(simulation)
            count +=1
        total += 1

    # Go back to ~/Trajectories
    os.chdir(working_dir)
print("{} out of {} simulations failed".format(count, total))
