#from __future__ import print_function
import os
import sys
import time

import pdb
import collections
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import itertools
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import pandas as pd
import mdtraj as mdtraj
import simtk.unit as unit

import grid_analysis
import group_templates

def block_avg(traj, data, block_size=5*unit.nanosecond):
    """
    Break a 2d numpy array into blocks

    This function is taken directly from Tim Moore's `block_avg` (see
    https://github.com/tcmoore3/block_avg).

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Used to determine timestep
    data : np.ndarray, shape=(m, n)
        The data to block; must be a 2-dimensional array
    block_size : simtk.unit.unit.Unit
        The size of each block, should have time dimension
    Returns
    -------
    blocks : np.ndarray, shape=(m/block_size, n)
        The block averaged data
    stds : np.ndarray, shape=(m/block_size, n)
        The standard deviation of each block
    Notes
    -----
    `m` must not necessarily be divisible by `block_size` ; in the case
    that it isn't, the data is trimmed *from the beginning* so that it is.
    """

    timestep = traj.time[1] * unit.picosecond - traj.time[0] * unit.picosecond
    block_size = int(block_size/timestep)
    remainder = data.shape[0] % block_size
    if remainder != 0:
        data = data[remainder:]
    n_blocks = int(data.shape[0] / block_size)
    data = data.reshape((n_blocks, block_size))
    blocks = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    return blocks, stds

def calc_APL(traj, n_lipid,blocked=False, block_size=5*unit.nanosecond):
    ''' 
    Input: Trajectory and number of lipids
    Compute areas by looking at x and y unit cell lengths
    Return: array of area per lipids (n_frame x 1) [Angstrom]
    '''
    area = unit.Quantity(traj.unitcell_lengths[:, 0] 
            * traj.unitcell_lengths[:, 1], unit.nanometer**2) # This is n_frame x 1
    area = area.in_units_of(unit.angstrom**2)
    if blocked:
        #timestep = traj.time[1] * unit.picosecond - traj.time[0] * unit.picosecond
        #block_size_frames = int(block_size/timestep)
        blocks, stds = block_avg(traj, area, block_size=5*unit.nanosecond)
        areastd = np.std(blocks)
        areaavg = np.mean(blocks)
    else:
        areaavg = np.mean(area)
        areastd = np.std(area)
    apl_list = area/(n_lipid/2)
    apl_avg = areaavg/(n_lipid/2)
    apl_std = areastd/(n_lipid/2)
    return (apl_avg, apl_std, apl_list)

def identify_groups(traj, forcefield='gromos53a6'):
    """ Identify tails and heads for all lipids in system

    Parameters
    ----------
    traj: MDTraj Trajectory
    forcefield: str, default 'gromos53a6'
        String describing the force field (see `group_templates.py`)

    Returns
    -------
    tail_groups : Dictionary mapping a lipidtail to its indices
        keys : residue index (with a or b if two-tailed)
        values : tail atom indices
    head_groups : Dictionary mapping residue names to respective list of indices
        keys : residue names
        values : headgroup atom indices 
        """
    ff_templates = {'gromos53a6': group_templates.gromos53a6_groups,
            'charmm36': group_templates.charmm36_groups}
    #if forcefield == 'gromos53a6':
    #    groups = group_templates.gromos53a6_groups()

    #else:
    #    sys.exit("Forcefield not supported")
    try:
        groups = ff_templates[forcefield]()
    except KeyError:
        sys.exit("Forcefield not supported")

    tail_groups = OrderedDict()
    head_groups = OrderedDict()
    # Initialize empty lists for each headgroup category
    for resname in list(set([residue.name for residue in traj.topology.residues])):
        if 'HOH' not in resname and 'SOL' not in resname and 'water' not in resname:
            head_groups[resname] = []

    # Iterate through each residue, finding the template associated with
    # the residue name, and shifting the indices based on the first atom index
    for residue in traj.topology.residues:
        if not residue.is_water:
            template = groups[residue.name]
            headgroups_i = [val + residue.atom(0).index for val in template['head']]
            head_groups[residue.name] += headgroups_i
            if "PC" in residue.name or "ISIS" in residue.name or 'cer' in residue.name:
                tail_1 = [val + residue.atom(0).index for val in template['tail_1']]
                tail_2 = [val + residue.atom(0).index for val in template['tail_2']]
                tail_groups[str(residue.index)+"a"] = tail_1
                tail_groups[str(residue.index)+"b"] = tail_2
            else:
                tail = [val + residue.atom(0).index for val in template['tail']]
                tail_groups[str(residue.index)] = tail

    return tail_groups, head_groups


def calc_tilt_angle(traj, topol, lipid_tails, blocked=False, 
        block_size=5*unit.nanosecond):
    ''' 
    Input: Trajectory, topology, dictionary of lipid tails with atom index values
    Compute characteristic vector using eigenvector associated with
    lowest eigenvalue of inertia tensor.
    Compute angle between charactersitic vector and lipid tail,
    adjusted for the first quadrant of  cartesian coordinate space
    Blocks of 5 ns (250 frames if 1 frame every 20 fs)
    Return: array of tilt angles (n_frame x n_lipid_tail)
    '''

    surface_normal = np.asarray([0, 0, 1.0])
    bot_leaflet, top_leaflet = identify_leaflets(traj) 

    mid_plane = np.mean([np.mean(traj.xyz[:,bot_leaflet,2]), 
                            np.mean(traj.xyz[:,top_leaflet,2])])
    angle_list = np.eye(traj.n_frames, len(lipid_tails.keys()))
    index = 0
    for key in lipid_tails.keys():
        lipid_i_atoms = lipid_tails[key]
        if abs(np.mean(traj.xyz[0, lipid_i_atoms,2]) - mid_plane) > 0.5:
            traj_lipid_i = traj.atom_slice(lipid_i_atoms)
            director = mdtraj.geometry.order._compute_director(traj_lipid_i)
            #lipid_angle = np.rad2deg(np.arccos(np.dot(director, surface_normal))) * unit.degree
            lipid_angle = np.rad2deg(np.arccos(np.dot(director, 
                                                    surface_normal))) 
            for i,angle in enumerate(lipid_angle):
                if angle >= 90:
                    angle = 180 - angle
                    lipid_angle[i] = angle
            #angle_list.append(lipid_angle)
            angle_list[:,index] = lipid_angle
            index += 1
    angle_list = unit.Quantity(angle_list, unit.degree)
    angle_frame_avg = np.mean(angle_list, axis = 1) # For each frame, average all tail tilt angles
    if blocked:
        blocks, stds = block_avg(traj, angle_frame_avg, block_size=5*unit.nanosecond)
        angle_std = np.std(blocks)
        angle_avg = np.mean(blocks)
        
    else:
        angle_avg = np.mean(angle_frame_avg)
        angle_std = np.std(angle_frame_avg)
    return angle_avg, angle_std, angle_list


def calc_APT(traj, apl_list, angle_list, n_tails_per_lipid, blocked=False):
    ''' Input: a matrix of area per lipids (each row is a frame               
        a matrix of tilt angels (each row is a frame, each column is a lipid)
        Return matrix of area per tail (n_frame x n_lipid_tail)
    '''
    # Each element in angle list correspond to a tail, and that element is a row of tilts per frame
    # Each element in apl list is the apl for a frame
    apt_list = angle_list
    apt_list = unit.Quantity(np.array([np.cos(
                            angle_list[i,:].in_units_of(unit.radian)._value) 
                            * apl_list[i]._value/n_tails_per_lipid  
                            for i, _ in enumerate(angle_list)]),
                            apl_list.unit)
    apt_frame_avg = unit.Quantity(np.mean(apt_list._value, axis = 1), 
                                apt_list.unit) # For each frame, averge all tail tilt angeles
    if blocked:
        blocks, stds = block_avg(traj, apt_frame_avg)
        apt_avg = np.mean(blocks)
        apt_std = np.std(blocks)#/(len(apt_block_avgs)**0.5)
    else: 
        apt_avg = np.mean(apt_list)
        apt_std = np.std(apt_list)
    return apt_avg, apt_std, apt_list


def calc_head_distance(traj, topol, head_indices, blocked=False):
    """
    Input: Trajectory, topology, indices of headgroup atoms
    For each frame, compute the average z-coordinate of the headgroup in the top and bot leaflet
    Compute the difference as the headgroup distance
    Return: array of headgroup distances (n_frame x 1)
    """
    mass_top = 0
    mass_bot = 0
    zcoord_top = 0
    zcoord_bot = 0
    atom_counter = 0
    bot_leaflet, top_leaflet = identify_leaflets(traj)
    mid_plane = np.mean([np.mean(traj.xyz[:,bot_leaflet,2]), np.mean(traj.xyz[:,top_leaflet,2])])
    for atom_j in head_indices:
        mass_i = get_mass(topol, atom_j)
        if abs(traj.xyz[0,atom_j,2] - mid_plane) > 0.5:
            if atom_j in top_leaflet:
                zcoord_top += mass_i * traj.atom_slice([atom_j]).xyz[:,0,2]
                mass_top += mass_i
            else:
                zcoord_bot += mass_i * traj.atom_slice([atom_j]).xyz[:,0,2]
                mass_bot += mass_i
    zcoord_top = zcoord_top / mass_top * unit.nanometer
    zcoord_bot = zcoord_bot / mass_bot * unit.nanometer
    head_dist_list = abs(zcoord_top - zcoord_bot).in_units_of(unit.angstrom)
    
    return head_dist_list

def compute_headgroup_distances(traj, topol, headgroup_dict, blocked=False):
    """
    Input: trajectory, topology, dictionary mapping molecule types to their headgroup indices
    For each molecule type, compute the distance between headgroups
    Return: dictionary of molecule types and their headgroup distances for each frame 
    Dictionary values are lists of (n_frame x 1)
    """
    headgroup_distance_dict = OrderedDict()
    for key in headgroup_dict.keys():
        headgroup_distance_dict[key] = calc_head_distance(traj, topol, headgroup_dict[key], blocked=False)
    return headgroup_distance_dict

def calc_bilayer_height(traj, headgroup_distance_dict,blocked=False,anchor='DSPC'):
    """
    Input:
    Dictionary of molecule types and their headroup distances
    Calculate bilayer height by comparing DSPC or DPPC headgroup distances
    anchor : str
        Reference group to look at height
    Return: bilayer height average, bilayer height std, bilayer heigh per frame list
    """
    dist_list = headgroup_distance_dict[anchor]
    if blocked:
        #dist_blocks = dist_list[:-1].reshape(int((traj.n_frames-1)/250),250)
        #dist_block_avgs = np.mean(dist_blocks,axis=1)
        blocks, stds = block_avg(traj, dist_list)
        dist_avg = np.mean(blocks)
        dist_std = np.std(blocks)
    else:
        dist_avg = np.mean(dist_list)
        dist_std = np.std(dist_list)


    return (dist_avg, dist_std, dist_list)

def calc_offsets(traj, headgroup_distance_dict, blocked=False, anchor="DSPC"):
    """ 
    Input:
    dictionary of molecule types and their headgroup distances
    anchor : str
       Reference group to compare offsets 
    Calculate the offsets with respect to the phosphate group
    Return: dictionary of offsets with respect to the phosphate group
    Values are (distance averaged over n_Frames, distance std over n_frame)
    """
    offset_dict = OrderedDict()
    
    for key in headgroup_distance_dict.keys():
        offset_list = (headgroup_distance_dict[anchor]- headgroup_distance_dict[key]) / 2
        if blocked:
            #offset_blocks = offset_list[:-1].reshape(int((traj.n_frames-1)/250),250)
            #offset_block_avgs = np.mean(offset_blocks, axis=1)
            blocks, stds = block_avg(traj, offset_list)
            offset_avg = np.mean(blocks)
            offset_std = np.std(blocks)
        else:
            offset_avg = np.mean(offset_list)
            offset_std = np.std(offset_list)
        offset_dict[key] = [offset_avg, offset_std]

    return offset_dict

def calc_nematic_order(traj, blocked=False, block_size=5*unit.nanosecond):
    """ COmpute nematic order over each leaflet"""

    bot_leaflet, top_leaflet, mid_plane = identify_leaflets(traj, 
                                                        return_mid_plane=True)
    top_chains = []
    bot_chains = []

    for i, residue in enumerate(traj.topology.residues):
        if not residue.is_water:
            atoms_i = [a.index for a in residue.atoms]
            if abs(np.mean(traj.xyz[:, atoms_i, 2]) - mid_plane) > 0.5:
                indices = [a.index for a in residue.atoms]
                if set(indices).issubset(set(top_leaflet)):
                    top_chains.append(indices)
                else:
                    bot_chains.append(indices)
    s2_top = mdtraj.compute_nematic_order(traj, indices=top_chains)
    s2_bot = mdtraj.compute_nematic_order(traj, indices=bot_chains)
    s2_list = (s2_top + s2_bot)/2
    if blocked:
        #s2_blocks = s2_list[:-1].reshape(int((traj.n_frames-1)/250),250)
        #s2_block_avgs = np.mean(s2_blocks, axis = 1)
        blocks, stds = block_avg(traj, s2_list, block_size=block_size)
        s2_ave = np.mean(blocks)
        s2_std = np.std(blocks)
    else:
        s2_ave = np.mean(s2_list)
        s2_std = np.std(s2_list)
    return s2_ave, s2_std, s2_list

def identify_leaflets(traj, return_mid_plane=False):
    """  Identify bilayer leaflets based on z coord
    """
    top_leaflet = []
    bot_leaflet = []
    all_z = []
    for residue in traj.topology.residues:
        if not residue.is_water:
            residue_atoms = [a.index for a in residue.atoms]
            for val in traj.atom_slice(residue_atoms).xyz[:,:,2].flatten():
                all_z.append(val)

    z_cutoff = np.mean(all_z)

    for residue in traj.topology.residues:
        if not residue.is_water:
            residue_atoms = [a.index for a in residue.atoms]
            mean_z = np.mean(traj.atom_slice(residue_atoms).xyz[:,:,2])
            if mean_z <= z_cutoff:
                for index in residue_atoms:
                    bot_leaflet.append(index)
            else:
                for index in residue_atoms:
                    top_leaflet.append(index)

    bot_leaflet = np.asarray(bot_leaflet).flatten()
    top_leaflet = np.asarray(top_leaflet).flatten()
    if return_mid_plane:
        return bot_leaflet, top_leaflet, z_cutoff

    return bot_leaflet, top_leaflet

def get_all_masses(traj, topol, atom_indices):
    """ Return array of masses corresponding to atom idnices"""
    masses = np.zeros_like(atom_indices, dtype=float)
    masses = []
    for i, index in enumerate(atom_indices):
        masses.append(get_mass(topol,index))
    masses = unit.Quantity(masses, unit=unit.gram)
    return masses


def calc_density_profile(traj, topol, bin_width=0.2, blocked=False, 
                        block_size=5*unit.nanosecond):
    """ Use numpy histogram, with weights, to get density profile"""
    bot_leaflet, top_leaflet = identify_leaflets(traj)
    area = unit.Quantity(np.mean(traj.unitcell_lengths[:, 0] 
                        * traj.unitcell_lengths[:, 1]), unit.nanometer*unit.nanometer)
    v_slice = area * unit.Quantity(bin_width*unit.nanometer)

    density_profile_bot = []
    density_profile_top = []
    density_profile_all = []
    bot_masses = (get_all_masses(traj, topol, bot_leaflet) / v_slice).in_units_of(unit.kilogram * (unit.meter**-3))
    top_masses = (get_all_masses(traj, topol, top_leaflet) / v_slice).in_units_of(unit.kilogram * (unit.meter**-3))
    bounds = (np.min(traj.xyz[:, bot_leaflet, 2]),
            np.max(traj.xyz[:, top_leaflet,2]))
    n_bins = int(round((bounds[1] - bounds[0]) / bin_width))
    bin_width = (bounds[1] - bounds[0]) / n_bins
    
    interdigitation = []
    for xyz in traj.xyz:
        bot_hist, bin_edges = np.histogram(xyz[bot_leaflet,2], bins=n_bins,
                range=bounds, normed=False, weights=bot_masses._value)
        top_hist, bin_edges = np.histogram(xyz[top_leaflet,2], bins=n_bins,
                range=bounds, normed=False, weights=top_masses._value)
        all_hist = [x+y for x,y in zip(top_hist, bot_hist)]
        density_profile_bot.append(bot_hist)
        density_profile_top.append(top_hist)
        density_profile_all.append(all_hist)

        bin_centers = bin_edges[1:] - bin_width / 2
        integrand = []
        for top_slice, bot_slice in zip(top_hist, bot_hist):
            numerator = 4 * top_slice * bot_slice
            denominator =  (top_slice + bot_slice) ** 2
            overlap = 0
            if denominator != 0:
                overlap = numerator / denominator
            integrand.append(overlap)
        interdigitation.append(integrate.simps(integrand, x = bin_centers))
    if blocked:
        blocks, stds = block_avg(traj, np.array(interdigitation), 
                                block_size=block_size)
        idig_avg = unit.Quantity(np.mean(blocks), unit.nanometer)
        idig_std = unit.Quantity(np.std(blocks), unit.nanometer)
    else:
        idig_avg = np.mean(interdigitation)
        idig_std = np.std(interdigitation)
    return np.asarray(density_profile_all), \
        np.asarray(density_profile_bot), np.asarray(density_profile_top), \
        bin_centers, interdigitation, idig_avg, idig_std


def get_mass(topol, atom_i):
    """
    Input: trajectory, atom index
    Mass dictionary is in units of amu (g/mol)
    Return: mass of that atom (g)
    """
    try:
        mass_dict = {'O': 15.99940, 'OM': 15.99940, 'OA': 15.99940, 'OE': 15.99940, 'OW': 15.99940,'N': 14.00670,
            'NT': 14.00670, 'NL': 14.00670, 'NR': 14.00670, 'NZ': 14.00670, 'NE': 14.00670, 'C': 12.01100, 
            'CH0': 12.0110, 'CH1': 13.01900, 'CH2': 14.02700, 'CH3': 15.03500, 'CH4': 16.04300, 'CH2r': 14.02700,
            'CR1': 13.01900, 'HC': 1.00800, 'H':  1.00800, 'P': 30.97380, 'CL': 35.45300, 'F': 18.99840, 
            'CL-': 35.45300}
        mass_i = (unit.Quantity(mass_dict[topol.atom(atom_i).name], unit.amu).in_units_of(unit.gram / unit.item)*unit.item)._value
    except KeyError:
        mass_i = (unit.Quantity(topol.atom(atom_i).element.mass, unit.amu).in_units_of(unit.gram /unit.item) * unit.item)._value
    return mass_i

def calc_interdigitation(traj, density_profile_top, density_profile_bot, bins, blocked=False):
    """
    Input: top and bottom density profile [kg/m3], but units irrelevant since this is dimensionless
    Compute interdigitation according to "Structural Properties.." by Hartkamp (2016)
    Densities based on nm bins, convert to Angstrom
    Return: Interdigation avg, std, and array of interdigation at each frame (n_frame x 1) [A]
    """
    interdig = integrate.simps( (4*density_profile_top*density_profile_bot)/
            ((density_profile_top + density_profile_bot)**2), x=bins)
    if blocked:
        interdig_blocks = interdig[:-1].reshape(int((traj.n_frames-1)/250), 250)
        interdig_block_avgs = np.mean(interdig_blocks,axis=1)
        interdig_avg = np.mean(interdig_block_avgs)
        interdig_std = np.std(interdig_block_avgs)
    else: 
        interdig_avg = np.mean(interdig)
        interdig_std = np.std(interdig)
    return interdig_avg, interdig_std, interdig

def calc_hbonds(traj, traj_pdb, topol, lipid_dict, headgroup_dict,include_water_solute=False):
    """ Compute hydrogen bonding between lipids and water
    
    Parameters
    ---------
    traj : mdtraj trajectory
    topol: mdtraj topology
    lipid_dict : dict
        Mapping residue indices to associated atom indices
    headgroup_dict : dict
        Mapping lipid type to atoms that comprise headgroups

    Returns
    ------
    Matrix whose elements correspond to the hydrogen bonds between the two groups

    Notes
    -----
    Using wernet-nilsson
    Baker-hubbard seems more appropriate for protein studies
    """

    # Identify which lipid types we're dealing with
    #lipid_type_atoms = OrderedDict()
    # Loop through the headgroup dict, each key is a lipid type
    # Construct label map to convert a lipid type into a numerical index for an array
    label_to_number = 0
    labelmap = OrderedDict()
    for lipid_type in headgroup_dict.keys():
        labelmap[lipid_type] = label_to_number
        label_to_number += 1
        

    # Add waters
    labelmap['HOH'] = label_to_number

    
    # Calc hbonds within a particular lipid type
    # Generic list to hold subsequent hbond matrices per frame
    hbond_matrix_list = []

    # Actual mdtraj computation of hbonds
    #hbonds = mdtraj.baker_hubbard(traj_pdb, exclude_water = True)
    hbonds = mdtraj.wernet_nilsson(traj_pdb, exclude_water = True, 
            include_water_solute=include_water_solute)

    # MDtraj generates a huge list of hyrogen bonds per frame
    for hbond_frame in hbonds:
        hbond_frame_matrix = np.zeros((len(labelmap.keys()), len(labelmap.keys())))
        # Interpret the hydrogen bond lists from mdtraj, sort into arrays of donors/acceptors
        for (atom_i, atom_j, atom_k) in hbond_frame:
            # Get the residues for each atom participating in a hbond
            # i is the donor atom, j is the hydrogen, k is the acceptor
            residue_i = topol.atom(atom_i).residue.name
            residue_j = topol.atom(atom_j).residue.name
            residue_k = topol.atom(atom_k).residue.name
            participating_residues = (residue_i, residue_j, residue_k)
            # Get residue names, convert them to indices for the matrix
            donor = labelmap[participating_residues[0]]
            try:
                acceptor = labelmap[participating_residues[2]]
            except KeyError:
                acceptor = labelmap['ISIS']
            hbond_frame_matrix[donor,acceptor]+=1
        # Add the frame's hbond matrix to the overall hbond matrix list
        hbond_matrix_list.append(hbond_frame_matrix)
    # Compute avgs and stds
    hbond_matrix_avg = np.mean(hbond_matrix_list, axis=0)
    hbond_matrix_std = np.std(hbond_matrix_list, axis=0)

    return (hbond_matrix_avg, hbond_matrix_std, hbond_matrix_list, labelmap)


def _compute_rotational_autocorrelation(traj, atom_1, atom_2, 
        dt = 10, n_time_origins=50):
    """ Compute rotational correlation over various time intervals
    For two atoms with respect to distance vector at a time origin

    Parameters
    ---------
    traj : mdtraj Trajectory
    atom_1 : int
        First atom index
    atom_2 : int
        Second atom index
    dt : int
        Time step (1 frame is dt ps)
    n_time_origins ; int

    Returns
    -------
    rotational_correlations = list()
        List of rotational correlations over each time interval
        """
    # Rot acfs is a list of lists
    # The first element is a list corresponding to rot acfs for time interval 0
    # The sec element is a list corresponding to rot acfs  for time interval 1

    rot_acfs = [[] for i in range(traj.n_frames)]


    # 50 time origins spaced evenly throughout trajectory
    time_origins = np.linspace(0, traj.n_frames-n_time_origins, num=n_time_origins)
    # Make sure these are integers though
    time_origins = [int(time_origin) for time_origin in time_origins]

    # For every time origin
    for time_origin in time_origins:
        # Compute the distance vector at time origin
        dist_vector_0 = [traj.xyz[time_origin, atom_1, 0] - traj.xyz[time_origin, atom_2, 0],
                        traj.xyz[time_origin, atom_1, 1] - traj.xyz[time_origin, atom_2, 1],
                        traj.xyz[time_origin, atom_1, 2] - traj.xyz[time_origin, atom_2,2]]
        # Reference vector is the vector at the time origin
        reference_vector = [dist_vector_0[0], dist_vector_0[1], 0]
        # Compute the cos(angle) at time origin
        cos_angle_0 = np.dot(reference_vector, dist_vector_0)/(np.dot(dist_vector_0, dist_vector_0)**0.5)
        # For every frame between origin and last frame
        for i in range(traj.n_frames):
    
            if i+time_origin < traj.n_frames:
                dist_vector_i = [traj.xyz[time_origin+i, atom_1, 0]-traj.xyz[time_origin+i, atom_2, 0],
                            traj.xyz[time_origin+i, atom_1, 1] - traj.xyz[time_origin+i, atom_2, 1],
                            traj.xyz[time_origin+i, atom_1, 2] - traj.xyz[time_origin+i, atom_2,2]]
                cos_angle_i = np.dot(reference_vector, dist_vector_i)/(np.dot(dist_vector_i, dist_vector_i)**0.5)
                # Calculate the autocorrelation and add to rot_acfs
                auto_corr = cos_angle_i * cos_angle_0 / (cos_angle_0**2)
                rot_acfs[i].append(auto_corr)
    
    # Average rot_acfs so each time interval has a single rot_acf
    correlation = [np.mean(rot_acf) for rot_acf in rot_acfs]

    return correlation

def compute_rotational_autocorrelation(traj,
        dt = 10, n_time_origins=5, forcefield='charmm36'):   
    """ Compute rotatoinal correlation over various time intervals
    For two atoms with respect to distance vector at a time origin

    Parameters
    ---------
    traj : mdtraj Trajectory
    atom_1 : int
        First atom index
    atom_2 : int
        Second atom index
    dt : int
        Time step (1 frame is dt ps)
    n_time_origins ; int

    Returns
    -------
    rotational_correlations = list()
        List of rotational correlations over each time interval
        """
    ff_templates = {'gromos53a6': group_templates.gromos53a6_groups,
            'charmm36': group_templates.charmm36_groups}
    #if forcefield == 'gromos53a6':
    #    groups = group_templates.gromos53a6_groups()

    #else:
    #    sys.exit("Forcefield not supported")
    try:
        groups = ff_templates[forcefield]()
    except KeyError:
        sys.exit("Forcefield not supported")

    # Need to figure out which moleucles are DSPC
    # [16,32] and [37,53] correspond to each tail (zero-index)
    all_correlations = []
    for resid, residue in enumerate(traj.topology.residues):
        if 'DSPC' in residue.name:
            local_atoms = [atom for atom in residue.atoms]
            #atom_1 = np.random.randint(16,33)
            #atom_2 = np.random.randint(37,54)
            atom_1 = np.random.choice(groups['DSPC']['tail_1'])
            atom_2 = np.random.choice(groups['DSPC']['tail_2'])
    
            global_1 = local_atoms[atom_1].index
            global_2 = local_atoms[atom_2].index
    
            correlation = _compute_rotational_autocorrelation(traj, global_1, global_2,
                    n_time_origins=n_time_origins, dt=dt)
            all_correlations.append(correlation)
        elif 'DPPC' in residue.name:
            local_atoms = [atom for atom in residue.atoms]
            #atom_1 = np.random.randint(16,30)
            #atom_2 = np.random.randint(35,49)
            atom_1 = np.random.choice(groups['DPPC']['tail_1'])
            atom_2 = np.random.choice(groups['DPPC']['tail_2'])

    
            global_1 = local_atoms[atom_1].index
            global_2 = local_atoms[atom_2].index
    
            correlation = _compute_rotational_autocorrelation(traj, global_1, global_2,
                    n_time_origins=n_time_origins, dt=dt)
            all_correlations.append(correlation)

    average_correlations = np.mean(all_correlations,axis=0)

    times = np.arange(0, (traj.n_frames)*dt, dt)
    return times, average_correlations


def compute_lateral_diffusion(traj, dt=20, n_time_origins=20):
    """ Compute xy diffusion
    Unwrap coordinates
    Compute MSDs
    Fit to diffusion

    Parameters
    ---------
    traj : mdtraj Trajecotry
    dt : int
        Timestep (ps)
    n_time_origins : int

    """
    # Frame-time conversions
    #times = np.arange(0, interval_max, dt)
    times = np.arange(0, (traj.n_frames)*dt, dt)
    # Space time origins evenly throughout trajectory
    # But make sure the last time origin still has space to calculate all time intervals
    time_origins = np.linspace(0, traj.n_frames-n_time_origins, num=n_time_origins)
    time_origins = [int(time_origin) for time_origin in time_origins]
    
    ### First step is unfolding the trajectory
    n_frames = traj.n_frames
    n_atoms = traj.topology.n_atoms
    non_water_residues = [res for res in traj.topology.residues if not res.is_water]
    n_residues = len(non_water_residues)
    unfolded_xyz = np.zeros((traj.n_frames, n_residues, 3))
    #unfolded_xyz_nocom = np.zeros((traj.n_frames, n_residues, 3))
    # Frame-by-frame set unfolded_xyz to be each residue's center of mass
    for resid, res in enumerate(non_water_residues):
        atom_indices = [atom.index for atom in res.atoms]
        unfolded_xyz[:, resid, :] = mdtraj.compute_center_of_mass(traj.atom_slice(atom_indices))
    
    ref_com = mdtraj.compute_center_of_mass(traj.atom_slice(atom_indices)[0])
    # Iterate through each frame, take frame 0 as reference
    # We start at frame 1 and look at the i-1 frame
    # This means the last frame has to be the last frame
    # Or n_frames-1. For ranges, this means [0, n_frames)
    start=time.time()
    for frame_index in np.arange(1, n_frames):
        frame_com = mdtraj.compute_center_of_mass(traj.atom_slice(atom_indices)[frame_index])
    
        # Look at each residue
        for resid, res_i in enumerate(non_water_residues):
    
            # Compare res_i.xyz @ frame_index and frame_index-1
            (dx, dy, dz) = unfolded_xyz[frame_index,resid,:] - unfolded_xyz[frame_index-1,resid,:]
    
            # Compute an average box between frame_index and frame_index-1
            avg_box_lengths = np.mean((traj.unitcell_lengths[frame_index],
                                        traj.unitcell_lengths[frame_index-1]),axis=0)
            
            # Scenario 1, big coordinate -> small coordinate (dx very negative)
            # Shift all subsequent coordinates up by box length
            if dx < -avg_box_lengths[0]/2:
                unfolded_xyz[frame_index:, resid,0] = [coord + avg_box_lengths[0] for coord in unfolded_xyz[frame_index:, resid, 0]]
            if dy < -avg_box_lengths[1]/2:
                unfolded_xyz[frame_index:, resid,1] = [coord + avg_box_lengths[1] for coord in unfolded_xyz[frame_index:, resid, 1]]
            if dz < -avg_box_lengths[2]/2:
                unfolded_xyz[frame_index:, resid,2] = [coord + avg_box_lengths[2] for coord in unfolded_xyz[frame_index:, resid, 2]]
            
            # Scenario 2, small coordinate -> big coordinate (dx very positive)
            # Shift all subsequent coordinates down by a box length
            if dx > avg_box_lengths[0]/2:
                unfolded_xyz[frame_index:, resid,0] = [coord - avg_box_lengths[0] for coord in unfolded_xyz[frame_index:, resid, 0]]
            if dy > avg_box_lengths[1]/2:
                unfolded_xyz[frame_index:, resid,1] = [coord - avg_box_lengths[1] for coord in unfolded_xyz[frame_index:, resid, 1]]
            if dz > avg_box_lengths[2]/2:
                unfolded_xyz[frame_index:, resid,2] = [coord - avg_box_lengths[2] for coord in unfolded_xyz[frame_index:, resid, 2]]

            # Also, shift the center of mass to the reference center of mass
            #unfolded_xyz_nocom[frame_index,resid,:] = unfolded_xyz[frame_index,resid,:] - frame_com + ref_com

    end=time.time()
    print("Unfolding: {}".format(end-start))
    
    
    # Now compute MSDs
    start=time.time()
    
    # Create a list
    # Each element is a list of squared deviations at that time interval
    all_sqdevs = [[] for i in range(traj.n_frames)]
    
    # Iterate through all time origins
    for time_origin in time_origins:
        
        # Get the unfolded coordinates at this time origin
        ref_xyz = unfolded_xyz[time_origin, :, :]
    
        # Iterate through all time intervals
        for dt in range(traj.n_frames):
            if dt + time_origin < traj.n_frames:
            
                # Look at a particular residue within this interval
                for resid, res_i in enumerate(non_water_residues):
    
                    # Gather the xy MSD for this one residue
                    deviation = [unfolded_xyz[time_origin+dt, resid, i] -
                            ref_xyz[resid, i] for i in range(2)]
    
                    sq_dev = np.sum([dev**2 for dev in deviation])
                    all_sqdevs[dt].append(sq_dev)
    all_msds = [np.mean(sqdev) for sqdev in all_sqdevs]
    end=time.time()
    print("multiple origins msd: {}".format(end-start))
    np.savetxt('msd.dat', np.column_stack((times,all_msds)))
    
    return times, all_msds

def fit_to_diffusion(independent_vars, dependent_vars, nd=3):
    """ Take a dataset and fit a line through it

    Parameters
    ---------
    independent_vars : list()
    dependent_vars: list()
    nd : int
        Number of dimensions (3 for 3d, 2 for 2d)

    """
    params, covar = curve_fit(line_func, independent_vars, dependent_vars) 
    return params[0]/(2*nd), params[1]

def line_func(x, slope, constant):
    """ Simple slope-intercept equation for line"""
    return (slope*x) + constant

def symmetrize(data, zero_boundary_condition=False):
    """Symmetrize a profile
    
    Params
    ------
    data : np.ndarray, shape=(n,)
        Data to be symmetrized
    zero_boundary_condition : bool, default=False
        If True, shift the right half of the curve before symmetrizing

    Returns
    -------
    dataSym : np.ndarray, shape=(n,)
        symmetrized data
    dataSym_err : np.ndarray, shape=(n,)
        error estimate in symmetrized data

    This function symmetrizes a 1D array. It also provides an error estimate
    for each value, taken as the standard error between the "left" and "right"
    values. The zero_boundary_condition shifts the "right" half of the curve 
    such that the final value goes to 0. This should be used if the data is 
    expected to approach zero, e.g., in the case of pulling a water molecule 
    through one phase into bulk water.
    """
    n_windows = data.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dataSym = np.zeros_like(data)
    dataSym_err = np.zeros_like(data)
    shift = {True: data[-1], False: 0.0}
    for i, sym_val in enumerate(dataSym[:n_win_half]):
        val = 0.5 * (data[i] + data[-(i+1)])
        err = np.std([data[i], data[-(i+1)] - shift[zero_boundary_condition]]) / np.sqrt(2)
        dataSym[i], dataSym_err[i] = val, err
        dataSym[-(i+1)], dataSym_err[-(i+1)] = val, err        
    return dataSym, dataSym_err

