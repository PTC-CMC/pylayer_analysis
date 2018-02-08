from __future__ import print_function
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
parser.add_option('-b', action='store_true', default = False, dest = 'blocked')

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
lipid_dict, headgroup_dict = bilayer_analysis_functions.get_lipids(topol)
lipid_tails,lipid_heads = bilayer_analysis_functions.get_lipid_tails(topol, lipid_dict)

n_lipid = len(lipid_dict.keys())
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
s2_ave, s2_std, s2_list = bilayer_analysis_functions.calc_nematic_order(traj, lipid_dict, blocked=options.blocked)
np.savetxt('s2.dat' s2_list)
print('Calculating headgroup distances...')
headgroup_distance_dict = bilayer_analysis_functions.compute_headgroup_distances(traj, topol, headgroup_dict, blocked=options.blocked)
print('Calculating bilayer height...')
Hpp_ave, Hpp_std, Hpp_list = bilayer_analysis_functions.calc_bilayer_height(traj, headgroup_distance_dict, blocked=options.blocked)
np.savetxt('height.dat', Hpp_list)
print('Calculating component offsets...')
offset_dict = bilayer_analysis_functions.calc_offsets(traj, headgroup_distance_dict, blocked=options.blocked)
print('Calculating density profile...')
d_a, d_t, d_b, bins, interdig_list,interdig_avg, interdig_std = \
    bilayer_analysis_functions.calc_density_profile(traj, topol, lipid_dict)
#print('Calculating hydrogen bonds...')
#hbond_matrix_avg, hbond_matrix_std, hbond_matrix_list, labelmap = bilayer_analysis_functions.calc_hbonds(traj, traj_pdb, topol, lipid_dict, headgroup_dict)

# Printing properties
print('Outputting to <{}>...'.format(outfilename))
outfile = open((outfilename + '.txt'),'w')
outpdf = PdfPages((outfilename+'.pdf'))
outfile.write('{:<20s}: {}\n'.format('Trajectory',trajfile))
outfile.write('{:<20s}: {}\n'.format('Structure',grofile))
outfile.write('{:<20s}: {}\n'.format('# Frames',traj.n_frames))
outfile.write('{:<20s}: {}\n'.format('Lipids',n_lipid))
outfile.write('{:<20s}: {}\n'.format('Tails',n_lipid_tails))
outfile.write('{:<20s}: {} ({})\n'.format('APL (A^2)',apl_avg, apl_std))
outfile.write('{:<20s}: {} ({})\n'.format('APT (A^2)',apt_avg, apt_std))
outfile.write('{:<20s}: {} ({})\n'.format('Bilayer Height (A)',Hpp_ave, Hpp_std))
outfile.write('{:<20s}: {} ({})\n'.format('Tilt Angle', angle_avg, angle_std))
outfile.write('{:<20s}: {} ({})\n'.format('S2', s2_ave, s2_std))
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
outfile.write('{:<20s}:\n'.format("Hbonding (D-A)"))
#for row_label in labelmap.keys():
#    for col_label in labelmap.keys():
#        row_index = labelmap[row_label]
#        col_index = labelmap[col_label]
#        hbond_avg = hbond_matrix_avg[row_index, col_index]
#        hbond_std = hbond_matrix_std[row_index, col_index]
#        outfile.write('{:<20s}: {} ({})\n'.format(str(row_label+"-"+ col_label), hbond_avg, hbond_std))


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
density_profile_average  = np.mean(d_a, axis=0)


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(bins,density_profile_avg)
plt.xlabel('Depth (nm)')
plt.title('Density Profile (kg m$^{-3}$)')


plt.subplot(2,1,2)

#plt.plot(bins,density_profile_bot_avg)
#plt.plot(bins,density_profile_top_avg)

plt.hist(np.mean(angle_list[:, 0 : int(np.floor(n_lipid_tails/2))], axis = 0), bins = 50,  
        alpha = 0.5, facecolor = 'blue', normed = True)
plt.hist(np.mean(angle_list[:, int(np.floor(n_lipid_tails/2)) : len(angle_list[0])], axis = 0), bins = 50,  
        alpha = 0.5, facecolor = 'red', normed = True)
plt.title('Angle Distribution by Leaflet')
plt.xlabel('Angle ($^o$)')

plt.tight_layout()
outpdf.savefig(fig2)
plt.close()
outpdf.close()

print('**********')
print('{:^10s}'.format('Done'))
print('**********')




