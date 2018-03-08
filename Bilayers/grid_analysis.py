import os
import time
import numpy as np
import pandas as pd
import itertools
import pdb
import glob
import subprocess
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import mdtraj
import permeability as prm
import bilayer_analysis_functions

""" Compute density profiles for lipids and water
compute density profiles for smaller XY grids
Compute local interfaces for XY regions as well as leaflet interfaces
Leaflet interfaces can be determined by water density or headgroup coordinates
Based on the 2D grid and definition of interface, compute tracer
permeability properties.
Save to pickle flie for later analysis

"""

def main():
    perm_data = pd.DataFrame()
    # If your'e in a sim folder
    #grid_analysis_routine()

    # Iterate through every sweep and sim directory
    curr_dir = os.getcwd()
    sweeps = [thing for thing in os.listdir() if 'sweep' in thing[0:6] and
            os.path.isdir(thing)]
    all_grid_outputs = []
    for sweep in sweeps:
        os.chdir(os.path.join(curr_dir, sweep))
        sims = [thing for thing in os.listdir() if 'Sim' in thing[0:4] and
            os.path.isdir(thing)]
        with Pool(5) as pool:
            grid_outputs = pool.starmap(_parallel_grid_analysis_routine, 
                    zip(itertools.repeat(curr_dir), itertools.repeat(sweep),
                        sims))
        for sim_output in grid_outputs:
            for tracer_dict in sim_output:
                perm_data = perm_data.append(tracer_dict, ignore_index=True)
    os.chdir(curr_dir)
    perm_data.to_pickle('perm_pickle.pkl')
        # Serial
        #for sim in sims:
        #    os.chdir(os.path.join(curr_dir, sweep, sim))
        #    grid_analysis_routine() 

def _parallel_grid_analysis_routine(curr_dir,sweep,sim):
    os.chdir(os.path.join(curr_dir, sweep,sim))
    tracer_outputs = grid_analysis_routine()
    return tracer_outputs


def grid_analysis_routine():
    curr_path = os.getcwd().split('/')[-2:]
    grofile = glob.glob("Stage4_*.gro")[0]
    trajfile = "trajectory.lammps"
    traj = mdtraj.load_lammpstrj(trajfile, top=grofile)
    # In case we have some false frames stuck on the front, just look at the
    # last 26 frames because that should be a continuous trjaectory
    traj = traj[-26:]
    traj = _wrap_trj(traj)
    grotraj = mdtraj.load(grofile)
    tracers = np.loadtxt('tracers.out', dtype=int)

    

    # Water density profile over a region
    water_indices = traj.topology.select('water') 
    headgroup_indices = _get_headgroup_indices(traj)
    water_p, bins = calc_density_profile(traj, traj.topology, water_indices,
            l_x=traj.unitcell_lengths[0,0],
            l_y=traj.unitcell_lengths[0,1])
    fig2=plt.figure(2)
    plt.plot(bins, water_p[0])
    plt.savefig("water_profile.jpg",transparent=True)
    plt.close()


    # Identification of top and bottom interfaces
    rho_water = 984 # SPC water
    #hho_interface = rho_water / np.e # Definition of water interface
    rho_interface = 100
    #z_interface_bot, z_interface_top = _find_interface_water(water_p, bins, rho_interface=rho_interface)
    z_interface_bot, z_interface_top = _find_interface_lipid(traj, headgroup_indices)
    leaflet_interfaces = [z_interface_bot, z_interface_top]


    # Generate 2D histogram of density 
    all_indices = [a.index for a in traj.topology.atoms if a.residue.is_water]
    thickness = 0.5
    grid_size = 1.5
    for i, z_interface in enumerate(leaflet_interfaces):
        if i == 0:
            name = "botwater"
        else:
            name = "topwater"

        # Given location of z_interface, find the atoms around it
        atoms_z = _find_atoms_around(traj, all_indices, thickness=thickness,
                z=z_interface)

        # Given that subset of atoms, then use the 2d histogram 
        density_surface, xbin_centers, ybin_centers, xedges, yedges = calc_density_surface(traj,
                atoms_z, grid_size=grid_size, thickness=thickness)

        _surface_plot(_normalize(density_surface), xbin_centers, ybin_centers,
                num_ticks=5, 
                title="Deviation from interfacial water density ({:.0f} kg/m$^3$)".format(rho_interface),
                filename='{}_waterdensity.jpg'.format(name))


    xbin_width = xbin_centers[1] - xbin_centers[0]
    ybin_width = ybin_centers[1] - ybin_centers[0]
    

    # Using the 2D hist bins, find the z-interface in each grid
    interface_bot_surface = np.zeros_like(density_surface)
    interface_top_surface = np.zeros_like(density_surface)

    for x, y in itertools.product(xbin_centers, ybin_centers):
        #atoms_xy = _find_atoms_within(traj, x=x, y=y, atom_indices=water_indices,
        atoms_xy = _find_atoms_within(traj, x=x, y=y, atom_indices=headgroup_indices,
                xbin_width=xbin_width, ybin_width=ybin_width)
        if len(atoms_xy) > 0:
            profile_xy, bins = calc_density_profile(traj, traj.topology, atoms_xy,
                    l_x=xbin_width, l_y=ybin_width)

            fig = plt.figure(1)
            plt.plot(bins, np.mean(profile_xy, axis=0))
            plt.xlabel("Z (nm)", fontsize=20)
            plt.ylabel("Density (kg/m$^3$)", fontsize=20)
            plt.savefig('{:.2f}_{:.2f}_waterp.jpg'.format(float(x), float(y)))
            plt.close()

            #z_interface_bot, z_interface_top = _find_interface_water(profile_xy, bins,
            #        rho_interface=rho_interface)
            z_interface_bot, z_interface_top = _find_interface_lipid(traj, atoms_xy)
            interface_bot_surface[ 0,int(np.floor(x/xbin_width)), 
                    int(np.floor(y/ybin_width))] = z_interface_bot
            interface_top_surface[ 0,int(np.floor(x/xbin_width)), 
                    int(np.floor(y/ybin_width))] = z_interface_top
        else:
            print("No atoms found around ({}, {})".format(x, y))
            interface_bot_surface[ 0,int(np.floor(x/xbin_width)), 
                    int(np.floor(y/ybin_width))] = -100
            interface_top_surface[ 0,int(np.floor(x/xbin_width)), 
                    int(np.floor(y/ybin_width))] = -100

    
    fig = plt.figure(1)
    plt.hist(_normalize(interface_bot_surface[0]).flatten(), 
            label="Bottom", alpha=0.4)
    plt.hist(_normalize(interface_top_surface[0]).flatten(), 
            label="Top", alpha=0.4)
    plt.xlabel("Interface location (mean set to 0) (nm)", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.legend()
    plt.savefig("Zinterface_histogram.jpg", transparent=True)
    plt.close()

            
    _surface_plot(_normalize(interface_bot_surface,reverse=True), xbin_centers, ybin_centers,
            num_ticks=5,
            title="Deviation from interface location (nm)",
            filename='interface_bot_surface.jpg')

    _surface_plot(_normalize(interface_top_surface), xbin_centers, ybin_centers,
            num_ticks=5,
            title="Deviation from interface location (nm)",
            filename='interface_top_surface.jpg')



    # Based on the xbins and ybins, figure out which tracers belong where
    all_tracer_outputs = []
    n_tracers = len(tracers)
    n_sims = _get_n_sims()

    for i, tracer in enumerate(tracers):
        # grid_output_dict will neatly contain all information and gets returned
        grid_output_dict = {}

        res = traj.topology.residue(tracer-1)
        tracer_oxygen = res.atom(0)
        xyz = traj.xyz[0, tracer_oxygen.index, :]

        # Identify which xy region we're in 
        bin_x = np.digitize(xyz[0], xedges) - 1
        bin_y = np.digitize(xyz[1], yedges) - 1

        # Identify the z interface for this xy region
        (interface_bot, interface_top) = (interface_bot_surface[0, bin_x, bin_y],
                interface_top_surface[0, bin_x, bin_y])

        # Find distance from local interface and leaflet interface
        if abs(interface_bot - xyz[2]) < abs(interface_top - xyz[2]):
            d_from_local_i = interface_bot - xyz[2]
            d_from_leaflet_i = leaflet_interfaces[0] - xyz[2]
            closest_interface = interface_bot
        else:
            d_from_local_i = xyz[2] - interface_top
            d_from_leaflet_i = xyz[2] = leaflet_interfaces[1] 
            closest_interface = interface_top

        # Find forceout file, do anallysis
        dz = 2 # Angstroms
        kB = 1.987e-3 # kcal/mol k
        T = 305
        RT2 = (kB*T)**2

        sim_number = int(curr_path[-1].replace('Sim', ''))
        forceout_index = sim_number + (i * n_sims)

        meanforce_file = '../meanforce{}.dat'.format(forceout_index)
        meanforce = np.loadtxt(meanforce_file)
        dG = -meanforce * dz

        fcorr_file = '../fcorr{}.dat'.format(forceout_index)
        int_F, int_F_val, FACF = prm.integrate_acf_over_time(fcorr_file, 
                timestep=1)

        diff_coeff = RT2 / int_F_val

        resist = np.exp(dG/ (kB * T)) / diff_coeff

        P = 1 / (resist * dz * 1e-8) # Convert z from \AA to cm


        grid_output_dict['leaflet_interfaces'] = leaflet_interfaces
        grid_output_dict['thickness'] = thickness
        grid_output_dict['grid_size'] = grid_size
        grid_output_dict['xbin_centers'] = xbin_centers
        grid_output_dict['ybin_centers'] = ybin_centers
        grid_output_dict['xedges'] = xedges
        grid_output_dict['yedges'] = yedges
        grid_output_dict['interface_bot'] = interface_bot_surface
        grid_output_dict['interface_top'] = interface_top_surface
        grid_output_dict['path'] = curr_path
        grid_output_dict['tracer'] = tracer
        grid_output_dict['local_interface'] = closest_interface
        grid_output_dict['d_from_local_i'] = d_from_local_i
        grid_output_dict['d_from_leaflet_i'] = d_from_leaflet_i
        grid_output_dict['meanforce_file'] = meanforce_file
        grid_output_dict['fcorr_file'] = fcorr_file
        grid_output_dict['free_energy'] = dG
        grid_output_dict['diffusion'] = diff_coeff
        grid_output_dict['permeability'] = P
        

        all_tracer_outputs.append(grid_output_dict)


        #all_G.append(dG)
        #all_D.append(diff_coeff)
        #all_P.append(P)


        #if interface_bot - 0.1 <= xyz[2] <= interface_bot + 0.1:
        #    sampled_interfaces.append(leaflet_interfaces[0] - interface_bot)
        #    analyze_tracer = True
        #elif interface_top - 0.1 <= xyz[2] <= interface_top + 0.1:
        #    sampled_interfaces.append(interface_top - leaflet_interfaces[1])
        #    analyze_tracer = True
        #else:
        #    analyze_tracer = False

        #if analyze_tracer:
        #    # Find forceout file
        #    n_tracers = len(tracers)
        #    n_sims = _get_n_sims()
        #    dz = 2 # Angstroms
        #    kB = 1.987e-3
        #    T = 305
        #    RT2 = (kB*T)**2

        #    sim_number = int(curr_path[-1].replace('Sim', ''))
        #    forceout_index = sim_number + (i * n_sims)

        #    meanforce_file = '../meanforce{}.dat'.format(forceout_index)
        #    meanforce = np.loadtxt(meanforce_file)
        #    dG = meanforce * dz

        #    fcorr_file = '../fcorr{}.dat'.format(forceout_index)
        #    int_F, int_F_val, FACF = prm.integrate_acf_over_time(fcorr_file, 
        #            timestep=1)

        #    diff_coeff = RT2 / int_F_val

        #    resist = np.exp(dG/ (kB * T)) / diff_coeff

        #    P = 1 / (resist * dz * 1e-8) # Convert z from \AA to cm

        #    all_G.append(dG)
        #    all_D.append(diff_coeff)
        #    all_P.append(P)




    return all_tracer_outputs

def _wrap_trj(traj):
    """ Wrap a trajectory """
    new_traj = mdtraj.Trajectory(xyz=traj.xyz, topology=traj.topology,
            time=traj.time, unitcell_lengths=traj.unitcell_lengths,
            unitcell_angles=traj.unitcell_angles)

    for i in range(new_traj.n_frames):
        for a in new_traj.topology.atoms:
            for j in range(3):
                while new_traj.xyz[i, a.index, j] < 0:
                    new_traj.xyz[i, a.index, j] += new_traj.unitcell_lengths[i, j]
                while new_traj.xyz[i, a.index, j] > new_traj.unitcell_lengths[i, j]:
                    new_traj.xyz[i, a.index, j] -= new_traj.unitcell_lengths[i, j]

                new_traj.xyz[i, a.index, j] = new_traj.xyz[i, a.index,j] \
                        + 0.5* new_traj.unitcell_lengths[i,j]

                while new_traj.xyz[i, a.index, j] < 0:
                    new_traj.xyz[i, a.index, j] += new_traj.unitcell_lengths[i, j]
                while new_traj.xyz[i, a.index, j] > new_traj.unitcell_lengths[i, j]:
                    new_traj.xyz[i, a.index, j] -= new_traj.unitcell_lengths[i, j]

    return new_traj




def get_mass(topol, atom_i):
    """
    Input: trajectory, atom index
    Mass dictionary is in units of amu
    Return: mass of that atom (g)
    """
    mass_dict = {'O': 15.99940, 'OM': 15.99940, 'OA': 15.99940, 'OE': 15.99940, 'OW': 15.99940,'N': 14.00670,
            'NT': 14.00670, 'NL': 14.00670, 'NR': 14.00670, 'NZ': 14.00670, 'NE': 14.00670, 'C': 12.01100, 
            'CH0': 12.0110, 'CH1': 13.01900, 'CH2': 14.02700, 'CH3': 15.03500, 'CH4': 16.04300, 'CH2r': 14.02700,
            'CR1': 13.01900, 'HC': 1.00800, 'H':  1.00800, 'P': 30.97380, 'CL': 35.45300, 'F': 18.99840, 
            'H2':  1.00800,'H1':  1.00800,
            'CL-': 35.45300}
    mass_i = 1.66054e-24 * mass_dict[topol.atom(atom_i).name]
    return mass_i

def get_all_masses(traj, topol, atom_indices):
    """ Return array of masses corresponding to atom idnices"""
    masses = np.zeros_like(atom_indices, dtype=float)
    for i, index in enumerate(atom_indices):
        masses[i] = get_mass(topol, index)
    return masses

def _get_headgroup_indices(traj):
    """ Return a giant list of all indices that correspond ot headgroups"""

    lipid_dict, headgroup_dict = bilayer_analysis_functions.get_lipids(traj.topology)
    headgroup_indices = []
    for key, val in headgroup_dict.items():
        for a in val:
            headgroup_indices.append(a)
    return headgroup_indices


def calc_density_profile(traj, topol, atom_indices, l_x=2, l_y=2,
        bin_width=0.2):
    """ Compute 1D density profile given a set of indices
    l_x : length of box, float (nm)
        This is used for defining the volume element 
    l_y : other length of box, float(nm)
        This is used for defining the volume element
    bin_width : bin widht for 1D profile, float (nm)
        This is the size of the bins for the 1 D profile
    
    
    """
    atoms = [a for a in traj.topology.atoms]
    desired_atoms = [atoms[i] for i in atom_indices]
    v_slice = l_x * l_y * bin_width
    density_profile = []
    sub_xyz = traj.xyz[:,atom_indices,:] 
    # Convert from g/nm3 to kg/m3, and nm to m
    #masses = 1e24 * get_all_masses(traj, topol, atom_indices) / v_slice
    masses = 1e24 * get_all_masses(traj, topol, atom_indices)

    # Find the absolute bounds over all frames

    bounds = (np.min(sub_xyz[:, :, 2]),
            np.max(sub_xyz[:, :, 2]))

    n_bins = int(round((bounds[1] - bounds[0]) / bin_width))
    bin_width = (bounds[1] - bounds[0]) / n_bins

    for xyz in sub_xyz:
        hist, edges = np.histogram(xyz[:,2], bins=n_bins,
                range=bounds, normed=False, weights=masses)
        density_profile.append(hist/v_slice)

        bin_centers = edges[1:] - bin_width / 2
    return np.asarray(density_profile), bin_centers

def calc_density_surface(traj, atom_indices, thickness=1,
        grid_size=0.2):
    """ Compute a density heatmap by gridding up space """

    atoms = [a for a in traj.topology.atoms]
    desired_atoms = [atoms[i] for i in atom_indices]
    sub_xyz = traj.xyz[:,atom_indices,:] 

    xbounds = (np.min(sub_xyz[:, :, 0]),
            np.max(sub_xyz[:, :, 0]))
    ybounds = (np.min(sub_xyz[:, :, 1]),
            np.max(sub_xyz[:, :, 1]))

    n_xbins = int(round((xbounds[1] - xbounds[0]) / grid_size))
    xbin_width = (xbounds[1] - xbounds[0]) / n_xbins

    n_ybins = int(round((ybounds[1] - ybounds[0]) / grid_size))
    ybin_width = (ybounds[1] - ybounds[0]) / n_ybins

    density_profile=[]
    v_slice = xbin_width * ybin_width * thickness
    #masses = 1e24 * get_all_masses(traj, traj.topology, atom_indices) / v_slice
    masses = 1e24 * get_all_masses(traj, traj.topology, atom_indices)


    for xyz in sub_xyz:
        hist, xedges, yedges = np.histogram2d(xyz[:,0], xyz[:,1],
                bins=[n_xbins, n_ybins], range=[xbounds, ybounds], 
                normed=False, weights=masses)
        density_profile.append(hist/v_slice)
        xbin_centers = xedges[1:] - xbin_width / 2
        ybin_centers = yedges[1:] - ybin_width / 2

    return np.array(density_profile), xbin_centers, ybin_centers, xedges, yedges


def _find_interface_water(water_p, bins, rho_interface=984,frame=0,reverse=False):
    """ Given a density profile of water,
    find the location of the interface
   
    reverse : boolean, default=False
        Used to iterate backwards through density profile,
        useful for finding the other interface
    """
    if reverse:
        step = 1
    else:
        step = -1
    # If using a time-averaged water profile:
    time_avg_water_p = np.mean(water_p, axis=0)
    midpoint = int(np.shape(time_avg_water_p)[0]/2)
    s = np.argsort(np.abs(time_avg_water_p[ : midpoint] - rho_interface))
    z_interface_bot = np.interp(rho_interface,
            [time_avg_water_p[s[0]], time_avg_water_p[s[1]], time_avg_water_p[s[2]]], 
            [bins[s[0]], bins[s[1]], bins[s[1]]])

    s = np.argsort(np.abs(time_avg_water_p[midpoint : ] - rho_interface))
    z_interface_top = np.interp(rho_interface,
            [time_avg_water_p[midpoint+s[0]], time_avg_water_p[ midpoint+s[1]], time_avg_water_p[ midpoint+s[2]]], 
            [bins[midpoint+s[0]], bins[midpoint+s[1]], bins[midpoint+ s[1]]])


    # If just using the first frame of the water profile
    #midpoint = int(np.shape(water_p)[1]/2)
    # Find closest indices to rho_interface
    #s = np.argsort(np.abs(water_p[0, : midpoint] - rho_interface))

    # Use linear interpolation to find more precise value of interface
        #z_interface_bot = np.interp(rho_interface,
    #        [water_p[0,s[0]], water_p[0,s[1]], water_p[0,s[2]]], 
    #        [bins[s[0]], bins[s[1]], bins[s[1]]])



    # Find closest indices to rho_interface
    #s = np.argsort(np.abs(water_p[0, midpoint : ] - rho_interface))

    # Use linear interpolation to find more precise value of interface
        #z_interface_top = np.interp(rho_interface,
    #        [water_p[0,midpoint+s[0]], water_p[0, midpoint+s[1]], water_p[0, midpoint+s[2]]], 
    #        [bins[midpoint+s[0]], bins[midpoint+s[1]], bins[midpoint+ s[1]]])

    return z_interface_bot, z_interface_top

def _find_interface_lipid(traj, headgroup_indices):
    """ Find the interface based on lipid head groups"""

    # Sort into top and bottom leaflet
    midplane = np.mean(traj.xyz[:,headgroup_indices,2])
    bot_leaflet = [a for a in headgroup_indices if traj.xyz[0,a,2] < midplane]
    top_leaflet = [a for a in headgroup_indices if traj.xyz[0,a,2] > midplane]

    com_bot = mdtraj.compute_center_of_mass(traj.atom_slice(bot_leaflet))
    com_top = mdtraj.compute_center_of_mass(traj.atom_slice(top_leaflet))

    z_interface_bot = np.mean(com_bot, axis=0)[2]
    z_interface_top = np.mean(com_top, axis=0)[2]

    return z_interface_bot, z_interface_top

def _find_atoms_within(traj, x=0, y=0, xbin_width=1, 
        ybin_width=1, atom_indices=None):
    """ Find atoms within a square with 
    center (x,y) and width 1, z irrelevant

    atom_indices: list of ints, optional
        If given a list of atom indices, only search through the given
        atom indices for those within the specific region
    """
    atom_search_space = [a for a in traj.topology.atoms]
    if atom_indices is not None:
        atom_search_space = [atom_search_space[i] for i in atom_indices]

    valid_indices = []
    for atom in atom_search_space:
        xyz = traj.xyz[0, atom.index, :]
        if (x - (xbin_width/2)) <= xyz[0] < (x + (xbin_width/2))  \
                and (y - (ybin_width/2)) <= xyz[1] < (y + (ybin_width/2)):
                    valid_indices.append(atom.index)

    return valid_indices

def _find_atoms_around(traj, atoms, thickness=1, z=0):
    """ Given a selection of atoms, find the ones that are 
    within `thickness` of `z` """

    valid_indices = []
    for atom in atoms:
        z_i = traj.xyz[0, atom, 2]
        if (z - (thickness / 2)) <= z_i < (z + (thickness / 2)):
            valid_indices.append(atom)
    return valid_indices

def _normalize(values, mean=None, reverse=False):
    """ Normalize values such that the mean is 0 ,
    
    mean : float, optional
        If specified, use this value and subtract `mean` from `values`
        If unspecified, compute the mean-per-frame to normalize
    reverse : bool, optional
        If true, swap the order of subtraction (useful for the lower leaflet)
        
        """

    new_array = np.zeros_like(values)
    if mean is None:
        for i in range(new_array.shape[0]):
            mean = np.mean(values[i])
            if reverse:
                new_array[i] = mean - values[i]
            else:
                new_array[i] = values[i] - mean

    return new_array


def _surface_plot(data, xbin_centers, ybin_centers,
        cmap='viridis', num_xticks=None, num_yticks=None, num_ticks=5,
        title="", filename=""):
        """ Some basic 2d surface plotting """
        # Yes we need to plot the transpose, otherwise the spatial X coords
        # get plotted on the Y axis and the Y coords get plotted on the X axis
        fig = plt.figure(1)
        plt.imshow(data[0,:,:].T, cmap=cmap, origin='lower')
        plt.colorbar()
        
        if num_xticks is None:
            num_xticks = num_ticks
        if num_yticks is None:
            num_yticks = num_ticks


        if data.shape[1] < num_xticks:
            num_xticks = data.shape[1]
        xtick_vals, step = np.linspace(0, data.shape[1], num=num_xticks, 
                dtype=int, retstep=True, endpoint=False)
        xtick_labels = [np.round(x,2) for x in xbin_centers][::int(np.floor(step))]
        plt.xticks(xtick_vals,  xtick_labels)

        if data.shape[2] < num_yticks:
            num_yticks = data.shape[2]
        ytick_vals, step = np.linspace(0, data.shape[2], num=num_yticks,
                dtype=int, retstep=True, endpoint=False)
        ytick_labels = [np.round(y,2) for y in ybin_centers][::int(np.floor(step))]
        plt.yticks(ytick_vals,  ytick_labels)

        plt.title(title)
        plt.savefig(filename, transparent=True)
        plt.close()

def _get_n_sims():
    working_dir = os.getcwd()
    os.chdir('..')
    sim_folders = [thing for thing in os.listdir() if 'Sim' in thing[0:4] and
            os.path.isdir(thing)]
    os.chdir(working_dir)
    return len(sim_folders)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Analysis took {}".format(end-start))

