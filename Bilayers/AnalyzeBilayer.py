from __future__ import print_function
import json
import sys
from optparse import OptionParser
import mdtraj
import pdb
import bilayer_analysis_functions 
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = OptionParser()
parser.add_option('-f', action="store", type="string", default = 'nopbc.xtc', dest = 'trajfile')
parser.add_option('-c', action="store", type="string", default = 'Stage5_ZCon0.gro', dest = 'grofile')
parser.add_option('-p', action="store", type="string", default = 'Stage5_ZCon0.gro', dest = 'pdbfile')
parser.add_option('-o', action='store', type='string', default = 'BilayerAnalysis', dest = 'outfilename')
parser.add_option('--nb', action='store_false', default=True, dest = 'blocked')

(options, args) = parser.parse_args()
trajfile = options.trajfile
grofile = options.grofile
pdbfile = options.pdbfile
outfilename = options.outfilename

print('Loading trajectory <{}>...'.format(trajfile))
print('Loading topology <{}>...'.format(grofile))
traj = mdtraj.load(trajfile, top=grofile)
print('Loading topology <{}>...'.format(pdbfile))
traj_pdb = mdtraj.load(trajfile, top=pdbfile)
topol = traj.topology

# Compute system information
print('Gathering system information <{}>...'.format(grofile))
lipid_tails, headgroup_dict = bilayer_analysis_functions.identify_groups(traj, 
        forcefield='charmm36')
n_lipid = len([res for res in traj.topology.residues if not res.is_water])
n_lipid_tails = len(lipid_tails.keys())
n_tails_per_lipid = n_lipid_tails/n_lipid



# Vectorized Calculations start here
print('Calculating area per lipid...')
apl_avg, apl_std, apl_list = bilayer_analysis_functions.calc_APL(traj,n_lipid, blocked=options.blocked)
np.savetxt('apl.dat', apl_list)
print('Calculating tilt angles...')
angle_avg, angle_std, angle_list = bilayer_analysis_functions.calc_tilt_angle(traj, topol, lipid_tails, blocked=options.blocked)
np.savetxt('angle.dat', angle_list)
print('Calculating area per tail...')
apt_avg, apt_std, apt_list = bilayer_analysis_functions.calc_APT(traj, apl_list, angle_list, n_tails_per_lipid, 
        blocked=options.blocked)
np.savetxt('apt.dat', apt_list)
print('Calculating nematic order...')
s2_ave, s2_std, s2_list = bilayer_analysis_functions.calc_nematic_order(traj, blocked=options.blocked)
np.savetxt('s2.dat', s2_list)
print('Calculating headgroup distances...')
headgroup_distance_dict = bilayer_analysis_functions.compute_headgroup_distances(traj, topol, headgroup_dict, blocked=options.blocked)
print('Calculating bilayer height...')
Hpp_ave, Hpp_std, Hpp_list = bilayer_analysis_functions.calc_bilayer_height(traj, headgroup_distance_dict, blocked=options.blocked, anchor='DSPC')
np.savetxt('height.dat', Hpp_list)
print('Calculating component offsets...')
offset_dict = bilayer_analysis_functions.calc_offsets(traj, headgroup_distance_dict, blocked=options.blocked, anchor='DSPC')
print('Calculating density profile...')
d_a, d_t, d_b, bins, interdig_list,interdig_avg, interdig_std = \
    bilayer_analysis_functions.calc_density_profile(traj, topol, 
                                                    blocked=options.blocked)
np.savetxt('idig.dat', interdig_list)
##print('Calculating hydrogen bonds...')
##hbond_matrix_avg, hbond_matrix_std, hbond_matrix_list, labelmap = bilayer_analysis_functions.calc_hbonds(traj, traj_pdb, topol, lipid_dict, headgroup_dict)
#
# Printing properties
print('Outputting to <{}>...'.format(outfilename))
datafile = OrderedDict()
datafile = OrderedDict()
datafile['trajectory'] = trajfile
datafile['structure'] = grofile
datafile['n_frames'] = traj.n_frames
datafile['lipids'] = n_lipid
datafile['tails'] = n_lipid_tails
datafile['APL'] = OrderedDict()
datafile['APL']['unit'] = str(apl_avg.unit)
datafile['APL']['mean'] = float(apl_avg._value)
datafile['APL']['std'] = float(apl_std._value)
datafile['APT'] = OrderedDict()
datafile['APT']['unit'] = str(apt_avg.unit)
datafile['APT']['mean'] = float(apt_avg._value)
datafile['APT']['std'] = float(apt_std._value)
datafile['Bilayer Height'] = OrderedDict()
datafile['Bilayer Height']['unit'] = str(Hpp_ave.unit)
datafile['Bilayer Height']['mean'] = float(Hpp_ave._value)
datafile['Bilayer Height']['std'] = float(Hpp_std._value)
datafile['Tilt Angle'] = OrderedDict()
datafile['Tilt Angle']['unit'] = str(angle_avg.unit)
datafile['Tilt Angle']['Bilayer'] = OrderedDict()
datafile['Tilt Angle']['Bilayer']['mean'] = float(angle_avg._value)
datafile['Tilt Angle']['Bilayer']['std'] = float(angle_std._value)
datafile['S2'] = OrderedDict()
datafile['S2']['mean'] = s2_ave
datafile['S2']['std'] = s2_std
datafile['Interdigitation'] = OrderedDict()
datafile['Interdigitation']['unit'] = str(interdig_avg.unit)
datafile['Interdigitation']['mean'] = float(interdig_avg._value)
datafile['Interdigitation']['std'] = float(interdig_std._value)

datafile['Offset'] = OrderedDict()
for key in offset_dict.keys():
    datafile['Offset']['unit'] = str(offset_dict[key][0].unit)
    datafile['Offset'][key] = OrderedDict()
    datafile['Offset'][key]['mean'] = float(offset_dict[key][0]._value )
    datafile['Offset'][key]['std'] = float(offset_dict[key][1]._value )
    #datafile['Offset (A)'][key] = [str(offset_dict[key][0]), str(offset_dict[key][1])]

datafile['Tilt Angle']['Leaflet 1'] = OrderedDict()
datafile['Tilt Angle']['Leaflet 1']['mean'] = float(np.mean(angle_list[:, 
                                        0 :int(np.floor(n_lipid_tails/2))])._value)
datafile['Tilt Angle']['Leaflet 1']['std'] = float(np.std(angle_list[:, 
                                            0 :int(np.floor(n_lipid_tails/2))])._value)

datafile['Tilt Angle']['Leaflet 2'] = OrderedDict()
datafile['Tilt Angle']['Leaflet 2']['mean'] = float(np.mean(angle_list[:, 
                                                int(np.floor(n_lipid_tails/2)):])._value)
datafile['Tilt Angle']['Leaflet 2']['std'] = float(np.std(angle_list[:, 
                                            int(np.floor(n_lipid_tails/2)):])._value)
#
with open(outfilename+'.txt', 'w') as f:
    json.dump(datafile, f, indent=2)

outpdf = PdfPages((outfilename+'.pdf'))

# Plotting

fig1 = plt.figure(1)
plt.subplot(3,2,1)
plt.plot(apl_list)
plt.title('APL')

plt.subplot(3,2,2)
plt.plot(np.mean(angle_list, axis=1))
plt.title('Tilt Angle ($^o$)')

plt.subplot(3,2,3)
plt.plot(np.mean(apt_list,axis=1))
plt.title('APT')

plt.subplot(3,2,4)
plt.plot(Hpp_list)
plt.title('H$_{PP}$')

plt.subplot(3,2,5)
plt.plot(s2_list)
plt.title('S2')

plt.subplot(3,2,6)
plt.plot(interdig_list)
plt.title('Interdigitation (A)')

plt.tight_layout()
outpdf.savefig(fig1)
plt.close()

density_profile_top_avg = np.mean(d_t, axis = 0)
density_profile_bot_avg = np.mean(d_b, axis = 0)
density_profile_avg  = np.mean(d_a, axis=0)
#
#
fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(bins,density_profile_avg)
plt.xlabel('Depth (nm)')
plt.title('Density Profile (kg m$^{-3}$)')


plt.subplot(2,1,2)

#plt.plot(bins,density_profile_bot_avg)
#plt.plot(bins,density_profile_top_avg)

plt.hist(np.mean(angle_list[:, 0 : int(np.floor(n_lipid_tails/2))], axis=0)._value, 
                bins=50,  
                alpha=0.5, facecolor='blue', normed=True)
plt.hist(np.mean(angle_list[:, int(np.floor(n_lipid_tails/2)) : len(angle_list[0])], 
                axis=0)._value, bins=50,  
                alpha=0.5, facecolor='red', normed = True)
plt.title('Angle Distribution by Leaflet')
plt.xlabel('Angle ($^o$)')

plt.tight_layout()
outpdf.savefig(fig2)
plt.close()
outpdf.close()


print('**********')
print('{:^10s}'.format('Done'))
print('**********')




