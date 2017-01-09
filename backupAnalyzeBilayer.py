from __future__ import print_function
import mdtraj as mdtraj
import sys
import os
from optparse import OptionParser
import pdb
import ipdb
import numpy as np
import matplotlib
import collections
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calc_APL(traj, n_lipid):
    ''' 
    Input: Trajectory and number of lipids
    Compute areas by looking at x and y unit cell lengths
    Return: array of area per lipids (n_frame x 1) [Angstrom]
    '''
    area = 100 * traj.unitcell_lengths[:, 0] * traj.unitcell_lengths[:, 1]
    areaavg = np.mean(area)
    areastd = np.std(area)/(len(area)**0.5)
    apl_list = np.eye(traj.n_frames, 1)
    apl_list[:,0] = area[:]/(n_lipid/2)
    apl_avg = areaavg/(n_lipid/2)
    apl_std = areastd/(n_lipid/2)
    return (apl_avg, apl_std, apl_list)

def get_lipids(topol):
    ''' Input a topology object
        Iterate through each atom
        If the atom's residue isn't water (so a lipid), add it to the dictionary
        lipid_dict is a dictionary mapping residue indices to a list of respective atom indices
    '''
    # Dictionary of resname keys that map to atom list values
    lipid_dict= dict()
    for i in topol.select('all'):
        atom_i = topol.atom(i)
        if not atom_i.residue.is_water:
            residue_i = atom_i.residue
            # If the lipid_dict already has the residue key, append it
            if residue_i.index in lipid_dict:
                lipid_dict[residue_i.index].append(i)
            # If the lipid_dict doesn't have the residue key, make a list and append it
            else:
                lipid_dict[residue_i.index] = list()
                lipid_dict[residue_i.index].append(i)
    return lipid_dict

def get_lipid_tails(topol, lipid_dict):
    ''' Input topology the lipid dictionary
        Look at the atoms in the lipid dictionary
        For each atom, get the index, (shifted to zero), residue name and residue index
        For that particular residue name, figure out if it belongs in a tail and add it
        If there are multiple tails for that residue, denote differences with 'a' and 'b'
        So Lipida and Lipidb are different keys in the lipid_tails dict
        But respective values correspond to that tail
        Return lipid_tails, a dictionary mapping each lipid tail to its atoms
    '''

    # Get an atom
    # Get that atom's residue
    # Shift atom indices by the index of the first atom in the residue 0
    # Based on the residue, check if that atom's shifted index falls within the tail range
    lipid_tails = dict()
    for lipid in lipid_dict.keys():
        lipid_atoms = lipid_dict[lipid]
        for atom_index in lipid_atoms:
            shifted_index = atom_index - lipid_atoms[0]
            atom_i = topol.atom(atom_index)
            resname = atom_i.residue.name
            resindex = atom_i.residue.index
            # This might need improvement, right now hard coding lipid tail definitions
            # Looking at 12 carbons after the headgroup
            if 'DSPC' in resname:
                if 14 == shifted_index or 16 <= shifted_index <= 26:
                    if (resname + str(resindex) + 'a') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'a')] = list()
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)

                elif 35 == shifted_index or 37 <= shifted_index <= 47:
                    if (resname + str(resindex) + 'b') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'b')] = list()
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)

            elif 'DPPC' in resname:
                if 14 == shifted_index or 16 <= shifted_index <= 26:
                    if (resname + str(resindex) + 'a') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'a')] = list()
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)

                elif 33 == shifted_index or 35 <= shifted_index <= 45:
                    if (resname + str(resindex) + 'b') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'b')] = list()
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)
            elif 'ISIS' in resname:
                if 6 <= shifted_index <= 17:
                    if (resname + str(resindex) + 'a') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'a')] = list()
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)
                elif 19 == shifted_index or 21 <= shifted_index <= 31:
                    if (resname + str(resindex) + 'b') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'b')] = list()
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)
            elif 'SS' in resname:
                if 5 <= shifted_index <= 16:
                    if (resname + str(resindex) + 'a') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'a')] = list()
                        lipid_tails[(resname + str(resindex) + 'a')].append(atom_index)
                elif 18 == shifted_index or 20 <= shifted_index <= 30:
                    if (resname + str(resindex) + 'b') in lipid_tails:
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex) + 'b')] = list()
                        lipid_tails[(resname + str(resindex) + 'b')].append(atom_index)

            elif 'acd16' in resname:
                if shifted_index == 16 or 4 <= shifted_index <= 14:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)

            elif 'acd22' in resname:
                if shifted_index == 22 or 10 <= shifted_index <= 20:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)

            elif 'alc12' in resname:
                if 0 <= shifted_index <= 11:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)

            elif 'alc14' in resname:
                if 0 <= shifted_index <= 13:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)

            elif 'alc16' in resname:
                if 0 <= shifted_index <= 15:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc18' in resname:
                if 0 <= shifted_index <= 17:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc20' in resname:
                if 0 <= shifted_index <= 19:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc22' in resname:
                if 0 <= shifted_index <= 21:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc24' in resname:
                if 0 <= shifted_index <= 23:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)

            else:
                print('Lipid {} not incorporated in lipid tail identification'.format(resname))
                sys.exit()
    return lipid_tails

def calc_tilt_angle(traj, topol, lipid_tails):
    ''' 
    Input: Trajectory, topology, dictionary of lipid tails with atom index values
    Compute characteristic vector using eigenvector associated with
    lowest eigenvalue of inertia tensor.
    Compute angle between charactersitic vector and lipid tail,
    adjusted for the first quadrant of  cartesian coordinate space
    Return: array of tilt angles (n_frame x n_lipid_tail)
    '''

    surface_normal = np.asarray([0, 0, 1.0])
    angle_list = []
    angle_list = np.eye(traj.n_frames, len(lipid_tails.keys()))
    index = 0
    for key in lipid_tails.keys():
        lipid_i_atoms = lipid_tails[key]
        traj_lipid_i = traj.atom_slice(lipid_i_atoms)
        director = mdtraj.geometry.order._compute_director(traj_lipid_i)
        lipid_angle = np.rad2deg(np.arccos(np.dot(director, surface_normal)))
        for i,angle in enumerate(lipid_angle):
            if angle >= 90:
                angle = 180- angle
                lipid_angle[i] = angle
        #angle_list.append(lipid_angle)
        angle_list[:,index] = lipid_angle
        index += 1
    angle_avg = np.mean(angle_list)
    angle_std = np.std(angle_list)/(len(angle_list)**0.5)
    return angle_avg, angle_std, angle_list


def calc_APT(apl_list, angle_list, n_tails_per_lipid):
    ''' Input: a matrix of area per lipids (each row is a frame               
        a matrix of tilt angels (each row is a frame, each column is a lipid)
        Return matrix of area per tail (n_frame x n_lipid_tail)
    '''
    # Each element in angle list correspond to a tail, and that element is a row of tilts per frame
    # Each element in apl list is the apl for a frame
    apt_list = angle_list
    apt_list = np.cos(np.deg2rad(angle_list[:,:]))*apl_list[:]/n_tails_per_lipid
    apt_avg = np.mean(apt_list)
    apt_std = np.std(apt_list)/(len(apt_list)**0.5)
    return apt_avg, apt_std, apt_list

def calc_mean(dataset):
    ''' Generic mean calculation 
    of a dataset. Assumes each elemtn in the dataset
    is a simtk Quantity'''

    avg = dataset[0]
    for i in range(1, len(dataset) - 1):
        avg = avg.__add__(dataset[i])
    avg = avg.__truediv__(len(dataset))
    return avg

def calc_stdev(avg, dataset):
    ''' Generic standard deviation calculation
    of a dataset. Assumes each element in the dataset
    is a simtk Quantity'''

    variance = (avg.__sub__(avg)).__pow__(2)
    for val in dataset:
        deviation =  val.__sub__(avg)
        variance = variance.__add__(deviation.__pow__(2))
    return (variance.__div__(len(dataset))).sqrt()

def read_xvg(filename):
    '''Given an xvg file, read the file
    Return the data as a list of lists
    Return the legend as a list '''

    xvgfile = open(filename, 'r')
    xvglines = xvgfile.readlines()
    data = list()
    legend = []
    for i, line in enumerate(xvglines):
        if '@' in line and 'legend' in line and 's' in line:
            first_apostrophe = line.find('\"')
            second_apostrophe = line.rfind('\"')
            legend_entry = line[first_apostrophe+1: second_apostrophe]
            legend_entry = legend_entry.replace('\\S', '$^{')
            legend_entry = legend_entry.replace('\\s', '$_{')
            legend_entry  = legend_entry.replace('\\N', '}$')
            legend.append(legend_entry)
        if '#' not in line and '@' not in line:
            items = line.split()
            data.append((items))
    
        else:
            pass
    return data, legend

def calc_phosphate_distance(traj, topol, phosphate_indices, n_phosphates):
    '''
    Input: trajectory, topology, indices of phosphate group atoms, number of phosphate groups
    For each frame, compute the average z-coordinate of phosphate groups in the top and bot leaflet
    Compute the difference as the phosphate distance
    Return: array of phosphate distances (n_frame x 1)
    '''
    mass_top = 0
    mass_bot = 0
    zcoord_top = 0
    zcoord_bot = 0
    atom_counter = 0  
    for atom_j in phosphate_indices:
        if 'P' in topol.atom(atom_j).name:
            mass_i = 30.97
        elif 'O' in topol.atom(atom_j).name:
            mass_i = 16.00

        if atom_counter < n_phosphates/2:
            zcoord_top += mass_i * traj.atom_slice([atom_j]).xyz[:,0,2]
            mass_top += mass_i
        else:
            zcoord_bot += mass_i * traj.atom_slice([atom_j]).xyz[:,0,2]
            mass_bot += mass_i
        atom_counter +=1
    zcoord_top = zcoord_top / mass_top
    zcoord_bot = zcoord_bot / mass_bot
    phosphate_dist_list = 10 * abs(zcoord_top - zcoord_bot)
    phosphate_dist_avg = np.mean(phosphate_dist_list)
    phosphate_dist_std = np.std(phosphate_dist_list)
    return phosphate_dist_avg, phosphate_dist_std, phosphate_dist_list


def calc_nematic_order(traj, lipid_dict):
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
    s2_ave = np.mean(s2_list)
    s2_std = np.std(s2_list)
    #s2_top_ave = np.mean(s2_top)
    #s2_bot_ave = np.mean(s2_bot)
    #s2_ave = (s2_top_ave + s2_bot_ave)/2
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

traj = mdtraj.load(trajfile, top=grofile)
topol = traj.topology
outfile = open((outfilename + '.txt'),'w')

# Compute system information
lipid_dict = get_lipids(topol)
lipid_tails = get_lipid_tails(topol, lipid_dict)

n_lipid = len(lipid_dict.keys())
n_lipid_tails = len(lipid_tails.keys())
n_tails_per_lipid = n_lipid_tails/n_lipid

phosphate_indices = topol.select("(symbol == P or name == OM or name == OA)")
n_phosphates = len(phosphate_indices)

# Vectorized Calculations start here
apl_avg, apl_std, apl_list = calc_APL(traj,n_lipid)
angle_avg, angle_std, angle_list = calc_tilt_angle(traj, topol, lipid_tails)
apt_avg, apt_std, apt_list = calc_APT(apl_list, angle_list, n_tails_per_lipid)
phosphate_dist_avg, phosphate_dist_std, phosphate_dist_list = calc_phosphate_distance(traj, topol, phosphate_indices, n_phosphates)
s2_ave, s2_std, s2_list = calc_nematic_order(traj, lipid_dict)

# Printing properties
outfile.write('Trajectory: {}\n'.format(trajfile))
outfile.write('Structure: {}\n'.format(grofile))
outfile.write('# Frames: {}\n'.format(traj.n_frames))
outfile.write('Lipids: {}\n'.format(n_lipid))
outfile.write('Tails: {}\n'.format(n_lipid_tails))
outfile.write('APL (A^2): {} ({})\n'.format(apl_avg, apl_std))
outfile.write('APT (A^2): {} ({})\n'.format(apt_avg, apt_std))
outfile.write('Bilayer Height (A): {} ({})\n'.format(phosphate_dist_avg, phosphate_dist_std))
outfile.write('Tilt Angle: {} ({})\n'.format(angle_avg, angle_std))
outfile.write('S2: {} ({})\n'.format(s2_ave, s2_std))

# Plotting
plt.subplot(3,2,1)
plt.plot(apl_list)
plt.title('APL')

plt.subplot(3,2,2)
plt.plot(np.mean(angle_list, axis=1))
plt.title('Tilt Angle')

plt.subplot(3,2,3)
plt.plot(np.mean(apt_list,axis=1))
plt.title('APT')

plt.subplot(3,2,4)
plt.plot(phosphate_dist_list)
plt.title('H$_{PP}$')

plt.subplot(3,2,5)
plt.plot(s2_list)
plt.title('S2')

plt.tight_layout()
plt.savefig((outfilename + '.pdf'), format='pdf')



