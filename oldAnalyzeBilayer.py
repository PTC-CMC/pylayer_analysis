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

def calc_box_info_frame(trajframe):
    next_xbox, next_ybox, next_zbox  = box_dimensions(trajframe)
    next_area = next_xbox.__mul__(next_ybox)
    return (next_xbox, next_ybox, next_zbox, next_area)
    
def calc_box_info(traj):
    # Input trajectory, return frame-averaged box infor
    xboxlist = []
    yboxlist = []
    zboxlist = []
    arealist = []
    for i in range(traj.n_frames):
        next_xbox, next_ybox, next_zbox  = box_dimensions(traj[i])
        next_area = next_xbox.__mul__(next_ybox)
        xboxlist.append(next_xbox)
        yboxlist.append(next_ybox)
        zboxlist.append(next_zbox)
        arealist.append(next_area)
    xboxavg = calc_mean(xboxlist)
    xboxstdev = calc_stdev(xboxavg, xboxlist)
    yboxavg = calc_mean(yboxlist)
    yboxstdev = calc_stdev(yboxavg, yboxlist)
    zboxavg = calc_mean(zboxlist)
    zboxstdev = calc_stdev(zboxavg, zboxlist)
    areaavg = calc_mean(arealist)
    areastdev = calc_stdev(areaavg, arealist)
    return (xboxavg, xboxstdev, yboxavg, yboxstdev,
            zboxavg, zboxstdev, areaavg, areastdev,
            arealist)

def box_dimensions(trajframe):
    ''' Input a trajectory frame
        Get the box vectors for frame
        Preserve units '''

    xbox = trajframe.openmm_boxes(0)[0][0]
    ybox = trajframe.openmm_boxes(0)[1][1]
    zbox = trajframe.openmm_boxes(0)[2][2]
    return xbox, ybox, zbox

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
                if 2 <= shifted_index <= 13:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)

            elif 'alc16' in resname:
                if 4 <= shifted_index <= 15:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc18' in resname:
                if 6 <= shifted_index <= 17:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc20' in resname:
                if 8 <= shifted_index <= 19:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc22' in resname:
                if 10 <= shifted_index <= 21:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)
            elif 'alc24' in resname:
                if 12 <= shifted_index <= 23:
                    if (resname + str(resindex)) in lipid_tails:
                        lipid_tails[(resname + str(resindex))].append(atom_index)
                    else:
                        lipid_tails[(resname + str(resindex))] = list()
                        lipid_tails[(resname + str(resindex))].append(atom_index)

            else:
                print('Lipid {} not incorporated in lipid tail identification'.format(resname))
                sys.exit()
    return lipid_tails


def calc_apt_angles(traj, lipid_tails):
    angle_list = []
    apt_list = []
    for i in range(traj.n_frames):
        angle = (calc_tilt_angle(lipid_tails, topol, traj[i]))
        apt = arealist[i].__truediv__(n_lipid_tails).__mul__(2*np.cos(np.deg2rad(angle)))
        angle_list.append(angle)
        apt_list.append(apt)
    
    tilt_angle = np.mean(angle_list)
    tilt_angle_stdev = np.std(angle_list)
    apt_avg = calc_mean(apt_list)
    apt_stdev = calc_stdev(apt_avg, apt_list)
    return (tilt_angle, tilt_angle_stdev, angle_list, apt_avg, apt_stdev, apt_list)


def calc_tilt_angle(lipid_dict, topol, trajframe):
    ''' Input a list of all the lipid atoms of one frame
        For each lipid chain, compute the moment of inertia tensor
        Dot that with the surface normal, then arccos
        Return a list of tilt angles for each lipid '''
    surface_normal = np.asarray([0, 0, 1.0])
    angle_list = []
    for key in lipid_dict.keys():
        lipid_i_atoms = lipid_dict[key]
        traj_lipid_i = trajframe.atom_slice(lipid_i_atoms)

        director = mdtraj.geometry.order._compute_director(traj_lipid_i)
        lipid_angle = np.rad2deg(np.arccos(np.dot(director, surface_normal)))[0]

        if lipid_angle >= 90:
            lipid_angle = 180- lipid_angle

        angle_list.append(lipid_angle)
    return np.mean(angle_list)

def calc_apl(arealist, n_lipid):
    apl_list = [x.__truediv__(n_lipid/2) for x in arealist]
    area_per_lipid = calc_mean(apl_list)
    apl_stdev = calc_stdev(area_per_lipid, apl_list)
    return (apl_list, area_per_lipid, apl_stdev)

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

def calc_mass_density_profile(trajfile, tprfile,  n_groups = 1, water_density = False, begin = 0, end = 0):
    ''' Use gmx density -n -s -f and -ng for number of groups
        return an xvg file with all the density proilfes on there
        density assumes z-direction'''
    if water_density:
        # With no components, water is group 2
        # with 3 components, water is group 5
        groups = 2 + n_groups
        os.system('echo {} | gmx density -f {} -s {} -sl 1000 '.format(groups, trajfile,
            tprfile))
    else:
    # Compute the groups we want
        groups = ''
        for i in range(2, 2+n_groups):
            groups = groups + ' ' + str(i)
        os.system('echo {} | gmx density -f {} -s {} -ng {} -sl 1000 -b {} -e {} '.format(groups, trajfile,
            tprfile, n_groups, begin, end))

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

def calc_bilayer_thickness_intervals(trajfile, tprfile, n_groups = 1, 
        water_density = False, tfinal = 100000, step  = 10000):
    bilayer_thickness_list = []
    for i in range(0, tfinal, step):
        calc_mass_density_profile(trajfile, tprfile, n_groups = n_groups, water_density = water_density, begin = i, end = i +step)
        data, legend = read_xvg('density.xvg')
        bilayer_thickness = calc_bilayer_thickness([item[0] for item in data], [item[1+legend.index('Water')] for item in data])
        bilayer_thickness_list.append(bilayer_thickness)
    bilayer_thickness_avg = np.mean(bilayer_thickness_list)
    bilayer_thickness_stdev = np.std(bilayer_thickness_list)
    return (bilayer_thickness_avg, bilayer_thickness_stdev)



def calc_bilayer_thickness(coordinates, densitylist):
    ''' Input the list of z coordinates and water densities
        Calculate water bulk density and where the density is 
        closeset to 1/e
        
    '''
    bulk_water = (float(densitylist[0]) + float(densitylist[-1]))/2
    midpoint = float(coordinates[-1])/2
    # errorlist contains a list of the two values closest to 1/e of bulk water
    # first index corresponds to top half, second index for bottom half
    errorlist = [9999, 9998]
    # zlist contains the z-coordinates of the two values closest to 1/e bulk water
    zlist = [0, 0]

    for i, density in enumerate(densitylist):
        error = abs(float(density) - (bulk_water/np.exp(1)))
        # Replace the error depending on which half of the bilayer we're on
        if error < errorlist[0] and float(coordinates[i]) < midpoint:
            zlist[0] = coordinates[i]
            errorlist[0] = error

        elif error < errorlist[1] and float(coordinates[i]) > midpoint:
            zlist[1] = coordinates[i]
            errorlist[1] = error

        else:
            pass
            
        
    return float(zlist[1]) - float(zlist[0])

def calc_phosphate_distance(trajframe, topol, phosphate_indices, n_phosphates):
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
            zcoord_top += mass_i * trajframe.atom_slice([atom_j]).xyz[0][0][2]
            mass_top += mass_i
        else:
            zcoord_bot += mass_i * trajframe.atom_slice([atom_j]).xyz[0][0][2]
            mass_bot += mass_i
        atom_counter +=1
    zcoord_top = zcoord_top / mass_top
    zcoord_bot = zcoord_bot / mass_bot
    phosphate_distance = abs(zcoord_top - zcoord_bot)
    return phosphate_distance

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
    s2_top_ave = np.mean(s2_top)
    s2_bot_ave = np.mean(s2_bot)
    s2_ave = (s2_top_ave + s2_bot_ave)/2
    return s2_ave

# CODE STARTS HERE
parser = OptionParser()
parser.add_option('-f', action="store", type="string", default = 'nopbc.xtc', dest = 'trajfile')
parser.add_option('-s', action="store", type="string", default = 'Stage5_ZCon0.tpr', dest = 'tprfile')
parser.add_option('-c', action="store", type="string", default = 'Stage5_ZCon0.gro', dest = 'grofile')
parser.add_option('--ng', action='store', type='int', default = 2, dest = 'n_groups')
parser.add_option('-t', action='store', type='int', default = 10, dest = 'tfinal')
parser.add_option('-o', action='store', type='string', default = 'BilayerAnalysis.txt', dest = 'outfilename')

(options, args) = parser.parse_args()
trajfile = options.trajfile
tprfile = options.tprfile
grofile = options.grofile
n_groups = options.n_groups
tfinal = options.tfinal
outfilename = options.outfilename

traj = mdtraj.load(trajfile, top=grofile)
topol = traj.topology
outfile = open(outfilename,'w')

print('# Frames: {}\n'.format(traj.n_frames))
outfile.write('# Frames: {}\n'.format(traj.n_frames))


# Basic topology information gathering
lipid_dict = get_lipids(topol)
lipid_tails = get_lipid_tails(topol, lipid_dict)

n_lipid = len(lipid_dict.keys())
n_lipid_tails = len(lipid_tails.keys())
n_tails_per_lipid = n_lipid_tails/n_lipid

phosphate_indices = topol.select("(symbol == P or name == OM or name == OA)")
n_phosphates = len(phosphate_indices)

xboxlist = []
yboxlist = []
zboxlist = []
arealist = []

angle_list = []
apt_list = []

phosphate_distance_list = []

# Compute properties frame by frame
for frame_i in range(traj.n_frames):
    print('Computing Frame: {}'.format(frame_i), flush=True)
    x_box, y_box, z_box, area_box = calc_box_info_frame(traj[frame_i])
    xboxlist.append(x_box)
    yboxlist.append(y_box)
    zboxlist.append(z_box)
    arealist.append(area_box)

    angle = (calc_tilt_angle(lipid_tails, topol, traj[frame_i]))
    apt = arealist[frame_i].__truediv__(n_lipid_tails).__mul__(2*np.cos(np.deg2rad(angle)))
    angle_list.append(angle)
    apt_list.append(apt)

    phosphate_distance = calc_phosphate_distance(traj[frame_i], topol, phosphate_indices, n_phosphates)
    phosphate_distance_list.append(phosphate_distance)


# Statistics come after information gathering
xboxavg = calc_mean(xboxlist)
xboxstdev = calc_stdev(xboxavg, xboxlist)
yboxavg = calc_mean(yboxlist)
yboxstdev = calc_stdev(yboxavg, yboxlist)
zboxavg = calc_mean(zboxlist)
zboxstdev = calc_stdev(zboxavg, zboxlist)
areaavg = calc_mean(arealist)
areastdev = calc_stdev(areaavg, arealist)

tilt_angle = np.mean(angle_list)
tilt_angle_stdev = np.std(angle_list)
apt_avg = calc_mean(apt_list)
apt_stdev = calc_stdev(apt_avg, apt_list)

phosphate_distance_avg = np.mean(phosphate_distance_list)
phosphate_distance_stdev = np.std(phosphate_distance_list)

# Post processing calculations
(apl_list, area_per_lipid, apl_stdev) = calc_apl(arealist, n_lipid)
s2_ave = calc_nematic_order(traj, lipid_dict)


# Printing stuff 
outfile.write('# Frames: {}\n'.format(traj.n_frames))
outfile.write('Box Dimensions:\n {} ({}) \n {} ({}) \n {} ({})\n'.format(xboxavg, xboxstdev,
    yboxavg, yboxstdev, zboxavg, zboxstdev))
outfile.write('Bilayer area: {} ({})\n'.format(areaavg, areastdev))
outfile.write('Lipids: {}\n'.format(n_lipid))
outfile.write('Tails: {}\n'.format(n_lipid_tails))
outfile.write('APL: {} ({})\n'.format(area_per_lipid, apl_stdev))
outfile.write('Tilt Angle: {} ({})\n'.format(tilt_angle, tilt_angle_stdev))
outfile.write('APT: {} ({})\n'.format(apt_avg, apt_stdev))
outfile.write('Bilayer Height: {} ({})\n'.format(phosphate_distance_avg, phosphate_distance_stdev))
outfile.write('S2: {}\n'.format(s2_ave))


''' THIS IS BAD BILAYER THICKNESS#
## Getting bilayer thickness, computing thickness every defined step
#(bilayer_thickness_avg, bilayer_thickness_stdev) = calc_bilayer_thickness_intervals(trajfile, 
#        tprfile, n_groups = n_groups, water_density = False, tfinal = tfinal, step = 1)
#
##print('Bilayer thickness (nm): {} ({})\n'.format(bilayer_thickness_avg,
#    #bilayer_thickness_stdev))
#
#outfile.write('Bilayer thickness (nm): {} ({})\n'.format(bilayer_thickness_avg,
#    bilayer_thickness_stdev))
#
'''
