import os
import pdb
import glob
import subprocess
from multiprocessing import Pool

import math
import numpy as np
import pandas as pd
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import mdtraj
import bilayer_analysis_functions
import grid_analysis

""" Iterate through all Data and Sim directories, 
assessing interfacial variation using gridding """

def main():
    root_dir = os.getcwd()
    to_pandas = []

    # Parallelization could occur here
    data_folders = [folder for folder in os.listdir() if os.path.isdir(folder)]
    for data_folder in data_folders:
        os.chdir(os.path.join(root_dir, data_folder))
        sim_folders = [folder for folder in os.listdir() if os.path.isdir(folder)]
        with Pool() as p:
            msr_avg_array = p.starmap(_parallel_routine, zip(itertools.repeat(root_dir), itertools.repeat(data_folder), sim_folders))
            for msr_avg in msr_avg_array:
                to_pandas.append([data_folder, msr_avg])
    # Do the analysis with al the mean squared roughness
    df = pd.DataFrame(data=to_pandas,index=None, columns=['Composition', 'Mean_Squared_Roughness'])
    compositions = sorted(list(set(df.loc[:,'Composition'])))
    for composition in compositions:
        tally = 0
        total_msr = 0
        for series in df.loc[df['Composition']==composition].values:
            if not math.isnan(series[1]):
                tally +=1
                total_msr += series[1]
        if tally > 0:
            avg_msr = total_msr / tally
        print(composition, avg_msr)

    pdb.set_trace() 

    #pre_analysis()

def _parallel_routine(root_dir, data_folder, sim_folder):
    return pre_analysis(path=os.path.join(root_dir,data_folder,sim_folder))

def pre_analysis(path=None):
    """ In a simulation folder, identify the mdtraj Trajectory """
    if path is not None:
        os.chdir(path)

    # Identify the proper files
    pdbfile = glob.glob("md*.pdb")
    xtcfile = glob.glob("last20.xtc")
    if len(pdbfile)>0 and len(xtcfile)>0:
        pdbfile = glob.glob("md*.pdb")[0]
        xtcfile = glob.glob("last20.xtc")[0]
        traj = mdtraj.load(xtcfile, top=pdbfile)
        msr_avg = analyze_simulation_interface(traj)
        return msr_avg

    else:
        return None

def analyze_simulation_interface(traj):
    """ Given an mdtraj Trajectory, identify the interface """
    # Identify all the headgroup indices 
    headgroup_indices = grid_analysis._get_headgroup_indices(traj)
    dspc_head_indices = [a for a in headgroup_indices
            if "DSPC" in traj.topology.atom(a).residue.name]

    # Identify the leaflet interfaces
    leaflet_interfaces = grid_analysis._find_interface_lipid(traj, headgroup_indices)

    # Grid up leafleats based on grid size
    grid_size = 2.0
    thickness = 0.5

    density_surface, xbin_centers, ybin_centers, xedges, yedges = \
            grid_analysis.calc_density_surface(traj, headgroup_indices, 
                    grid_size=grid_size, thickness=thickness)
    xbin_width = xbin_centers[1] - xbin_centers[0]
    ybin_width = ybin_centers[1] - ybin_centers[0]

    # Iterate through each grid point to find local interfaces
    b_interface_surface = np.zeros((len(xbin_centers), len(ybin_centers)))
    t_interface_surface = np.zeros((len(xbin_centers), len(ybin_centers)))
    for x, y in itertools.product(xbin_centers, ybin_centers):
        atoms_xy = grid_analysis. _find_atoms_within(traj, x=x, y=y, 
                atom_indices=headgroup_indices, 
                xbin_width=xbin_width, ybin_width=ybin_width)

        if len(atoms_xy) > 0:
            local_interfaces = grid_analysis._find_interface_lipid(traj, atoms_xy)
            b_interface_surface[int(np.floor(x/xbin_width)), 
                     int(np.floor(y/ybin_width))] = local_interfaces[0]
            t_interface_surface[int(np.floor(x/xbin_width)), 
                     int(np.floor(y/ybin_width))] = local_interfaces[1]

    # Normalize the surfaces based on the leaflet interface
    b_roughness = grid_analysis._normalize(b_interface_surface, reverse=True,
            mean=leaflet_interfaces[0])
    t_roughness = grid_analysis._normalize(t_interface_surface,
            mean=leaflet_interfaces[1])

    # Compute mean squared roughness (MSR)
    msr_b = np.sum(b_roughness**2)/b_roughness.size
    msr_t = np.sum(t_roughness**2)/t_roughness.size
    msr_avg = (msr_b + msr_t) / 2
    return msr_avg

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
        

