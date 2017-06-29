import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess

# Navigate through all the composition and simulation folders
# Find the occupied_area.dat files and average all of them
with open("safe_systems.txt",'r') as f:
    safe_systems = [line.strip() for line in f]

working_dir = os.getcwd()
for line in open("systems_of_interest.txt", 'r'):
#for line in open("testing2.txt", 'r'):
    composition_folder = line.strip()
    print(composition_folder)
    os.chdir(os.path.join(working_dir, composition_folder))
    simulation_folders = [f for f in os.listdir() if os.path.isdir(f)]
    occupied_areas = []
    simulation_names = []
    # Looping through each simulation folder
    for simulation in simulation_folders: 
        os.chdir(os.path.join(working_dir, composition_folder, simulation))
        print(simulation)
        if "occupied_area.dat" in os.listdir() and simulation in safe_systems:
            temp = np.loadtxt("occupied_area.dat")

            # Before adding this profile, center the minimum on 5
            # First find the locations of the two peaks, 
            # And the minimum will lie between the two peaks
            # Otherwise the minimum will (of course) be 0 outside bilayer
            midpoint_index = np.shape(temp)[0]/2
            peak_1 = np.argmax(temp[0:midpoint_index,1])
            peak_2 = peak_1 + np.argmax(temp[midpoint_index:,1])
            min_index = peak_1 + np.argmin(temp[peak_1 : peak_2, 1])
            shift_profile = 5 - temp[min_index, 0]
            
            #import pdb
            #pdb.set_trace()

            temp[:,0] += shift_profile

            occupied_areas.append(temp)
            simulation_names.append(simulation)

    # Compute the average occupied area profile
    average_occupied_area = np.mean(occupied_areas, axis=0)
    std_occupied_area = np.std(occupied_areas, axis=0)
    
    whole_occupied_area = np.hstack((average_occupied_area, std_occupied_area))
    os.chdir(os.path.join(working_dir, composition_folder))
    np.savetxt("average_occupied_area.dat", whole_occupied_area)

    # Plot it
    fig, axarray = plt.subplots(2,1)
    axarray[0].plot(whole_occupied_area[:,0], whole_occupied_area[:,1], 'k-', label=composition_folder)
    axarray[0].fill_between(whole_occupied_area[:,0], whole_occupied_area[:,1] 
            - whole_occupied_area[:,3], whole_occupied_area[:,1] 
            + whole_occupied_area[:,3], alpha=0.25, facecolor='cornflowerblue')
    axarray[0].set_ylim([0 ,1])
    axarray[0].set_xlim([0, 10])

    axarray[0].legend(loc='best')
    
    # alternative plot
    for single_profile, simulation in zip(occupied_areas ,simulation_names):
        axarray[1].plot(single_profile[:,0], single_profile[:,1], 
                label=simulation.split("_")[-1])
    axarray[1].set_ylim([0 ,1])
    axarray[1].set_xlim([0, 10])

    axarray[1].legend(loc='best')


    plt.savefig('average_occupied_area.png')
    plt.close()
    p = subprocess.Popen("cp average_occupied_area.png ~/Trajectories/Data/occ_profiles/{}.png".format(composition_folder), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()

