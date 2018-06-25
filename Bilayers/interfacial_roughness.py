import os
import pdb
import glob
import subprocess
from multiprocessing import Pool

import math
import numpy as np
import json
import pandas as pd
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import mdtraj
import simtk.unit as u
import bilayer_analysis_functions
import grid_analysis

############
## Functions to analyze bilayer-water interface from a simulation
########


def main():
    index = json.load(open('index.txt' ,'r'))
    curr_dir = os.getcwd()
    with Pool(16) as p:
            msr_pool = p.starmap(rough_routine, zip(itertools.repeat(curr_dir),index.keys()))

    df = pd.DataFrame(data=msr_pool, index=None, columns=['name', 'MSR_mean', 'MSR_std'])
    os.chdir(curr_dir)
    df.to_csv('roughness.csv')


def rough_routine(root_dir, sim_folder):
    """ In a simulation folder, identify the mdtraj Trajectory """
    os.chdir(os.path.join(root_dir, sim_folder))

    # Identify the proper files
    if os.path.isfile('npt.gro') and os.path.isfile('npt_80-100ns.xtc'):
        traj = mdtraj.load('npt_80-100ns.xtc', top='npt.gro')
        msr_results = analyze_simulation_interface(traj)
        msr_results['name'] = sim_folder
        return msr_results

    else:
        return None

def analyze_simulation_interface(traj, return_variance=False):
    """ Given an mdtraj Trajectory, identify the interface """
    # Identify all the headgroup indices 
    headgroup_indices = grid_analysis._get_headgroup_indices(traj)
    dspc_head_indices = [a for a in headgroup_indices
            if "DSPC" in traj.topology.atom(a).residue.name]

    

    # Grid up leafleats based on grid size
    grid_size = 2.0
    thickness = 0.5

    density_surface, xbin_centers, ybin_centers, xedges, yedges = \
            grid_analysis.calc_density_surface(traj, headgroup_indices, 
                    grid_size=grid_size, thickness=thickness)
    xbin_width = xbin_centers[1] - xbin_centers[0]
    ybin_width = ybin_centers[1] - ybin_centers[0]

    # Iterate through each frame
    # Identify the leaflet interfaces
    msr_list = []
    variance_list = []
    for i, frame in enumerate(traj):
        leaflet_interfaces = grid_analysis._find_interface_lipid(frame, 
                                                            headgroup_indices,
                                                            return_variance=False)
        # Iterate through each grid point to find local interfaces
        grid_msr = []
        grid_variance =[]
        for x, y in itertools.product(xbin_centers, ybin_centers):
            atoms_xy = grid_analysis._find_atoms_within(frame, x=x, y=y, 
                    atom_indices=headgroup_indices, 
                    xbin_width=xbin_width, ybin_width=ybin_width)

            if len(atoms_xy) > 0:
                if return_variance:
                    bot, top, variance  = grid_analysis._find_interface_lipid(
                                                    frame, 
                                                    atoms_xy,
                                                    return_variance=return_variance)
                else:
                    bot, top  = grid_analysis._find_interface_lipid(
                                                    frame, 
                                                    atoms_xy,
                                                    return_variance=return_variance)


                # Normalize the surfaces based on the leaflet interface
                if bot and top and all(leaflet_interfaces):
                    b_roughness = -1*(bot - leaflet_interfaces[0])
                    t_roughness = top - leaflet_interfaces[1]
                    msr = (np.sqrt(b_roughness**2) + np.sqrt(t_roughness**2)) / 2
                    grid_msr.append(msr)
                    if return_variance:
                        grid_variance.append(variance)
        if all(grid_msr):   
            msr_list.append(np.mean(grid_msr))
            if return_variance:
                variance_list.append(np.mean(grid_variance))
            #b_roughness = grid_analysis._normalize(local_interfaces[0], reverse=True,
            #        mean=leaflet_interfaces[i][0])
            #t_roughness = grid_analysis._normalize(local_interfaces[1],
            #        mean=leaflet_interfaces[i][1])

    # Compute mean squared roughness (root-MSR)
    np.savetxt('MSR.dat', np.asarray(msr_list))
    blocks, stds = bilayer_analysis_functions.block_avg(traj, np.asarray(msr_list), block_size=5*u.nanosecond)
    msr_avg = np.mean(blocks)
    msr_std = np.std(blocks)

    np.savetxt('z_variance.dat', np.asarray(variance_list))
    blocks, stds = bilayer_analysis_functions.block_avg(traj, np.asarray(variance_list), block_size=5*u.nanosecond)
    variance_avg = np.mean(blocks)
    variance_std = np.std(blocks)

    return {'MSR_mean':msr_avg, 'MSR_std':msr_std, 
            'z_variance_mean': variance_avg, 'z_variance_std': variance_std}

    # Plotting
    #_surface_plot(b_roughness, xbin_centers, ybin_centers, num_ticks=5,
    #        title="Deviation from leaflet interface", 
    #        filename="local_interface_bot.jpg")
    #_surface_plot(t_roughness, xbin_centers, ybin_centers, num_ticks=5,
    #        title="Deviation from leaflet interface", 
    #        filename="local_interface_top.jpg")


    #return leaflet_interfaces, b_interface_surface, t_interface_surface



def _surface_plot(data, xbin_centers, ybin_centers,
        cmap='viridis', num_xticks=None, num_yticks=None, num_ticks=5,
        vmin=-0.2, vmax=0.2,
        title="", filename=""):
        """ Some basic 2d surface plotting """
        # Yes we need to plot the transpose, otherwise the spatial X coords
        # get plotted on the Y axis and the Y coords get plotted on the X axis
        fig = plt.figure(1)
        # Construct normalization colormap, normalized clormap
        plt.imshow(data[:,:].T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar()
        
        if num_xticks is None:
            num_xticks = num_ticks
        if num_yticks is None:
            num_yticks = num_ticks


        if data.shape[0] < num_xticks:
            num_xticks = data.shape[0]
        xtick_vals, step = np.linspace(0, data.shape[0], num=num_xticks, 
                dtype=int, retstep=True, endpoint=False)
        xtick_labels = [np.round(x,2) for x in xbin_centers][::int(np.floor(step))]
        plt.xticks(xtick_vals,  xtick_labels)

        if data.shape[1] < num_yticks:
            num_yticks = data.shape[1]
        ytick_vals, step = np.linspace(0, data.shape[1], num=num_yticks,
                dtype=int, retstep=True, endpoint=False)
        ytick_labels = [np.round(y,2) for y in ybin_centers][::int(np.floor(step))]
        plt.yticks(ytick_vals,  ytick_labels)

        plt.title(title)
        plt.savefig(filename, transparent=True)
        plt.close()





if __name__ == "__main__":
    main()
        

