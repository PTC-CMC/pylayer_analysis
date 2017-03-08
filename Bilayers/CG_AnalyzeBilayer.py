from __future__ import print_function
import mdtraj as mdtraj
import sys
import scipy.integrate as integrate
import os
from optparse import OptionParser
import pdb
import ipdb
import numpy as np
import matplotlib
import collections
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def calc_APL(traj, n_lipid):
    ''' 
    Input: Trajectory and number of lipids
    Compute areas by looking at x and y unit cell lengths
    Return: array of area per lipids (n_frame x 1) [Angstrom]
    '''
    area = 100 * traj.unitcell_lengths[:, 0] * traj.unitcell_lengths[:, 1] # This is n_frame x 1
    #area_blocked = area[:-1].reshape(int((traj.n_frames-1)/250), 250) # Reshape so that each row is a block of 250 frames (5ns)
   # area_block_avg = np.mean(area_blocked, axis=1)
    area_block_avg = np.mean(area)
    areaavg = np.mean(area_block_avg)
    #areastd = np.std(area_block_avg)#/(len(area_block_avg)**0.5)
    areastd = np.std(area)#/(len(area_block_avg)**0.5)   
    apl_list = np.eye(traj.n_frames, 1)
    apl_list[:,0] = area[:]/(n_lipid/2)
    apl_avg = areaavg/(n_lipid/2)
    apl_std = areastd/(n_lipid/2)
    return (apl_avg, apl_std, apl_list)

def get_lipids(topol):
    """Gather lipid and headgroup information for system

    Parameters
    ---------
    topol : mdTraj topology

    Returns
    ------
    lipid_dict : dict
        dictionary mapping key values of residue indices to atom indices
    headggroup_dict : dict
        dictionary mapping key values of molecule names to headgroup atom indices
        """

    lipid_dict = OrderedDict()
    headgroup_dict = OrderedDict()
    for i in topol.select('all'):
        atom_i = topol.atom(i)
        residue_i = atom_i.residue
        resname = atom_i.residue.name
        # If this is isn't a water bead/atom then we do stuff
        if 'W' not in resname:
            # If the lipid_dict already has the residue key, append it
            if residue_i.index in lipid_dict:
                lipid_dict[residue_i.index].append(i)
            # If the lipid_dict doesn't have the residue key, make a list and append it
            else:
                lipid_dict[residue_i.index] = list()
                lipid_dict[residue_i.index].append(i)
            # Figure out headgroups
            if 'DSPC' in resname:
                if 'PO4' in atom_i.name:
                    if 'DSPC' in headgroup_dict:
                        headgroup_dict['DSPC'].append(i)
                    else:
                        headgroup_dict['DSPC'] = list()
                        headgroup_dict['DSPC'].append(i)
            elif 'DPPC' in resname:
                if 'PO4' in atom_i.name:
                    if 'DPPC' in headgroup_dict:
                        headgroup_dict['DPPC'].append(i)
                    else:
                        headgroup_dict['DPPC'] = list()
                        headgroup_dict['DPPC'].append(i)   
            elif 'C12OH' in resname:
                if 'OH' in atom_i.name:
                    if 'C12OH' in headgroup_dict:
                        headgroup_dict['C12OH'].append(i)
                    else:
                        headgroup_dict['C12OH'] = list()
                        headgroup_dict['C12OH'].append(i)
            elif 'C14OH' in resname:
                if 'OH' in atom_i.name:
                    if 'C14OH' in headgroup_dict:
                        headgroup_dict['C14OH'].append(i)
                    else:
                        headgroup_dict['C14OH'] = list()
                        headgroup_dict['C14OH'].append(i)
            elif 'C16OH' in resname:
                if 'OH' in atom_i.name:
                    if 'C16OH' in headgroup_dict:
                        headgroup_dict['C16OH'].append(i)
                    else:
                        headgroup_dict['C16OH'] = list()
                        headgroup_dict['C16OH'].append(i)
            elif 'C18OH' in resname:
                if 'OH' in atom_i.name:
                    if 'C18OH' in headgroup_dict:
                        headgroup_dict['C18OH'].append(i)
                    else:
                        headgroup_dict['C18OH'] = list()
                        headgroup_dict['C18OH'].append(i)
            elif 'C20OH' in resname:
                if 'OH' in atom_i.name:
                    if 'C20OH' in headgroup_dict:
                        headgroup_dict['C20OH'].append(i)
                    else:
                        headgroup_dict['C20OH'] = list()
                        headgroup_dict['C20OH'].append(i)
            elif 'C22OH' in resname:
                if 'OH' in atom_i.name:
                    if 'C22OH' in headgroup_dict:
                        headgroup_dict['C22OH'].append(i)
                    else:
                        headgroup_dict['C22OH'] = list()
                        headgroup_dict['C22OH'].append(i)
            elif 'C24OH' in resname:
                if 'OH' in atom_i.name:
                    if 'C24OH' in headgroup_dict:
                        headgroup_dict['C24OH'].append(i)
                    else:
                        headgroup_dict['C24OH'] = list()
                        headgroup_dict['C24OH'].append(i)
            elif 'C16FFA' in resname:
                if 'COO' in atom_i.name:
                    if 'C16FFA' in headgroup_dict:
                        headgroup_dict['C16FFA'].append(i)
                    else:
                        headgroup_dict['C16FFA'] = list()
                        headgroup_dict['C16FFA'].append(i)
            elif 'C22FFA' in resname:
                if 'COO' in atom_i.name:
                    if 'C22FFA' in headgroup_dict:
                        headgroup_dict['C22FFA'].append(i)
                    else:
                        headgroup_dict['C22FFA'] = list()
                        headgroup_dict['C22FFA'].append(i)

    return lipid_dict, headgroup_dict

def calc_head_distance(traj, topol, head_indices):
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
    for atom_j in head_indices:
        if 'PO4' in topol.atom(atom_j).name:
            mass_i = 94.97
        elif 'OH' in topol.atom(atom_j).name:
            mass_i = 17.01
        elif 'COO' in topol.atom(atom_j).name:
            mass_i = 44.01
        if atom_counter < len(head_indices)/2:
            zcoord_top += mass_i * traj.atom_slice([atom_j]).xyz[:,0,2]
            mass_top += mass_i
        else:
            zcoord_bot += mass_i * traj.atom_slice([atom_j]).xyz[:,0,2]
            mass_bot += mass_i
        atom_counter +=1
    zcoord_top = zcoord_top / mass_top
    zcoord_bot = zcoord_bot / mass_bot
    head_dist_list = 10 * abs(zcoord_top - zcoord_bot)
    #head_dist_blocks = head_dist_list[:-1].reshape(int((traj.n_frames-1)/250),250)
    #head_dist_block_avgs = np.mean(head_dist_blocks, axis = 1)
    head_dist_block_avgs = np.mean(head_dist_list)
    head_dist_avg = np.mean(head_dist_block_avgs)
    #head_dist_std = np.std(head_dist_block_avgs)
    head_dist_std = np.std(head_dist_list)
    return head_dist_avg, head_dist_std, head_dist_list

def compute_headgroup_distances(traj, topol, headgroup_dict):
    """
    Input: trajectory, topology, dictionary mapping molecule types to their headgroup indices
    For each molecule type, compute the distance between headgroups
    Return: dictionary of molecule types and their headgroup distances for each frame 
    Dictionary values are lists of (n_frame x 1)
    """
    headgroup_distance_dict = OrderedDict()
    for key in headgroup_dict.keys():
        headgroup_distance_dict[key] = calc_head_distance(traj, topol, headgroup_dict[key])
    return headgroup_distance_dict

def calc_bilayer_height(headgroup_distance_dict):
    """
    Input: Dictionary of molecule types and their headroup distances
    Calculate bilayer height by comparing DSPC or DPPC headgroup distances
    Return: bilayer height average, bilayer height std, bilayer heigh per frame list
    """
    if headgroup_distance_dict['DSPC']:
        dist_list = headgroup_distance_dict['DSPC'][2]
        #dist_frame_avg = np.mean(dist_list, axis = 1)
        #dist_blocks = dist_list[:-1].reshape(int((traj.n_frames-1)/250),250)
       # dist_block_avgs = np.mean(dist_blocks,axis=1)
        dist_block_avgs = np.mean(dist_list)
        dist_avg = np.mean(dist_block_avgs)
        #dist_std = np.std(dist_block_avgs)
        dist_std = np.std(dist_list)
    elif headgroup_distance_dict['DPPC']:
        dist_list = headgroup_distance_dict['DPPC'][2]
        #dist_frame_avg = np.mean(dist_list, axis = 1)
        #dist_blocks = dist_list[:-1].reshape(int((traj.n_frames-1)/250),250)
        #dist_block_avgs = np.mean(dist_blocks, axis=1)
        dist_block_avgs = np.mean(dist_list)
        dist_avg = np.mean(dist_block_avgs)
        #dist_std = np.std(dist_block_avgs)
        dist_std = np.std(dist_list)
    else:
        print ('No phosphate groups to compare')
    return dist_avg, dist_std, dist_list

def calc_offsets(headgroup_distance_dict):
    """ 
    Input: dictionary of molecule types and their headgroup distances
    Calculate the offsets with respect to the phosphate group
    Return: dictionary of offsets with respect to the phosphate group
    Values are (distance averaged over n_Frames, distance std over n_frame)
    """
    offset_dict = OrderedDict()
    if headgroup_distance_dict['DSPC']:
        use_DSPC = True
        use_DPPC = False
    elif headgroup_distance_dict['DPPC']:
        use_DSPC = False
        use_DPPC = True
    else:
        print('No phosphate groups to compare to')
    for key in headgroup_distance_dict.keys():
        # Determine if reference is DSPC or DPPC
        if use_DSPC:
            offset_list = (headgroup_distance_dict['DSPC'][2] - headgroup_distance_dict[key][2])/2
            #offset_frame_avg = np.mean(offset_list, axis = 1)
            #offset_blocks = offset_list[:-1].reshape(int((traj.n_frames-1)/250),250)
            #offset_block_avgs = np.mean(offset_blocks, axis = 1)
            offset_block_avgs = np.mean(offset_list)
            offset_avg = np.mean(offset_block_avgs)
            #offset_std = np.std(offset_block_avgs)
            offset_std = np.std(offset_list)
            offset_dict[key] = (offset_avg, offset_std)
        elif use_DPPC:
            offset_list = (headgroup_distance_dict['DPPC'][2] - headgroup_distance_dict[key][2])/2
            #offset_frame_avg = np.mean(offset_list, axis = 1)
            #offset_blocks = offset_list[:-1].reshape(int((traj.n_frames-1)/250),250)
            #offset_block_avgs = np.mean(offset_blocks, axis = 1)
            offset_block_avgs = np.mean(offset_list)
            offset_avg = np.mean(offset_block_avgs)
            #offset_std = np.std(offset_blocks_avgs)
            offset_std = np.std(offset_list)
            offset_dict[key] = (offset_avg, offset_std)
        else:
            print ('No phosphate groups to compare')

    return offset_dict

def calc_nematic_order(traj, lipid_dict):
    """Compute nematic order parameter
    
    Parameters
    ----------
    traj : mdTraj traj()
    
    lipid_dict : dict()
        dictionary apping residue indices to associated atom indices

    Returns
    -------
    s2_ave : float
        Average S2 over all frames
    s2_std : 
        S2 standard dviation
    s2_list :
        Frame by frame list of S2 values

    Notes
    -----
    Assumes 64 residues per leaflet
        """
    top_chains = []
    bot_chains = []
    for i, key in enumerate(lipid_dict.keys()):
        indices = [int(item) for item in lipid_dict[key]]
        if i <= 63:
            top_chains.append(indices)
        else:
            bot_chains.append(indices)
    s2_top = mdtraj.compute_nematic_order(traj, indices=top_chains)
    s2_bot = mdtraj.compute_nematic_order(traj, indices=bot_chains)
    s2_list = (s2_top + s2_bot)/2
    #s2_blocks = s2_list[:-1].reshape(int((traj.n_frames-1)/250),250)
    #s2_block_avgs = np.mean(s2_blocks, axis = 1)
    s2_block_avgs = np.mean(s2_list)
    s2_ave = np.mean(s2_block_avgs)
    #s2_std = np.std(s2_block_avgs)
    s2_std = np.std(s2_list)
    return s2_ave, s2_std, s2_list


# CODE STARTS HERE
parser = OptionParser()
parser.add_option('-f', action="store", type="string", default = 'nopbc.xtc', dest = 'trajfile')
parser.add_option('-c', action="store", type="string", default = 'Stage5_ZCon0.gro', dest = 'grofile')
parser.add_option('-o', action='store', type='string', default = 'BilayerAnalysis', dest = 'outfilename')

(options, args) = parser.parse_args()
trajfile = options.trajfile
grofile = options.grofile
outfilename = options.outfilename

print('Loading trajectory <{}>...'.format(trajfile))
traj = mdtraj.load(trajfile, top=grofile)
topol = traj.topology


# Compute system information
print('Gathering system information <{}>...'.format(grofile))
lipid_dict, headgroup_dict = get_lipids(topol)
#lipid_tails = get_lipid_tails(topol, lipid_dict)

n_lipid = len(lipid_dict.keys())
#n_lipid_tails = len(lipid_tails.keys())
#n_tails_per_lipid = n_lipid_tails/n_lipid


# Vectorized Calculations start here
print('Calculating area per lipid...')
apl_avg, apl_std, apl_list = calc_APL(traj,n_lipid)

#print('Calculating tilt angles...')
#angle_avg, angle_std, angle_list = calc_tilt_angle(traj, topol, lipid_tails)
#print('Calculating area per tail...')
#apt_avg, apt_std, apt_list = calc_APT(apl_list, angle_list, n_tails_per_lipid)
print('Calculating nematic order...')
s2_ave, s2_std, s2_list = calc_nematic_order(traj, lipid_dict)
print('Calculating headgroup distances...')
headgroup_distance_dict = compute_headgroup_distances(traj, topol, headgroup_dict)
print('Calculating bilayer height...')
Hpp_ave, Hpp_std, Hpp_list = calc_bilayer_height(headgroup_distance_dict)
print('Calculating component offsets...')
offset_dict = calc_offsets(headgroup_distance_dict)
#print('Calculating density profile...')
#density_profile, density_profile_avg, density_profile_top, density_profile_bot, bins = \
#    calc_density_profile(traj, topol, lipid_dict)
#print('Calculating interdigitation...')
#interdig_avg, interdig_std, interdig_list = calc_interdigitation(traj, density_profile_top, density_profile_bot, bins)

# Printing properties

print('Outputting to <{}>...'.format(outfilename))
outfile = open((outfilename + '.txt'),'w')
outpdf = PdfPages((outfilename+'.pdf'))
outfile.write('{:<20s}: {}\n'.format('Trajectory',trajfile))
outfile.write('{:<20s}: {}\n'.format('Structure',grofile))
outfile.write('{:<20s}: {}\n'.format('# Frames',traj.n_frames))
outfile.write('{:<20s}: {}\n'.format('Lipids',n_lipid))
#outfile.write('{:<20s}: {}\n'.format('Tails',n_lipid_tails))
outfile.write('{:<20s}: {} ({})\n'.format('APL (A^2)',apl_avg, apl_std))
#outfile.write('{:<20s}: {} ({})\n'.format('APT (A^2)',apt_avg, apt_std))
outfile.write('{:<20s}: {} ({})\n'.format('Bilayer Height (A)',Hpp_ave, Hpp_std))
#outfile.write('{:<20s}: {} ({})\n'.format('Tilt Angle', angle_avg, angle_std))
outfile.write('{:<20s}: {} ({})\n'.format('S2', s2_ave, s2_std))
"""
outfile.write('{:<20s}: {} ({})\n'.format('Interdigitation (A)', interdig_avg, interdig_std))

for key in offset_dict.keys():
    outfile.write('{:<20s}: {} ({})\n'.format
            ((key + ' offset (A)'), offset_dict[key][0], offset_dict[key][1]))
outfile.write('{:<20s}: {} ({})\n'.format(
    'Leaflet 1 Tilt Angle', np.mean(angle_list[:, 0 :int(np.floor(n_lipid_tails/2))]),
    np.std(angle_list[:, 0 :int(np.floor(n_lipid_tails/2))])))
outfile.write('{:<20s}: {} ({})\n'.format(
    'Leaflet 2 Tilt Angle', np.mean(angle_list[:, int(np.floor(n_lipid_tails/2)):len(angle_list[0])]), 
    np.std(angle_list[:, int(np.floor(n_lipid_tails/2)):len(angle_list[0])])))
"""

