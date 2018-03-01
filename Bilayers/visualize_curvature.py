import numpy as np
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mdtraj
import bilayer_analysis_functions

""" 2D Animation of lipid headgroup centers of masses
colors reflect Z coordinate,
the animation function should work. If it doesn't, then your ffmpeg 
or matplotlib is out of date"""

def _animate(i, my_plot, all_xyz,all_colors,mymap):

    my_plot.set_offsets(all_xyz[i,:, 0:2])
    my_plot.set_facecolor(all_colors[i])
    
    return my_plot,

grofile = 'centered.gro'
trajfile ='Stage1_Weak0.xtc'
#trajfile ='short.xtc'
traj = mdtraj.load(trajfile, top=grofile)
#traj = mdtraj.load(grofile)

topol = traj.topology
lipid_dict, headgroup_dict = bilayer_analysis_functions.get_lipids(topol)
lipid_tails,lipid_heads = bilayer_analysis_functions.get_lipid_tails(topol, lipid_dict)

bot_leaf, top_leaf = bilayer_analysis_functions.identify_leaflets(traj, topol, lipid_dict)

# This code will get centers of masses of all lipid head groups
all_xyz  = []
    
#for frame in range(10):
for frame in range(traj.n_frames):
    leaf_residues = list(set([topol.atom(a).residue for a in top_leaf]))

    frame_xyz = []
    for res in leaf_residues:
        if not res.is_water:
            [x,y,z] = mdtraj.compute_center_of_mass(traj.atom_slice(lipid_heads[str(res.index)]))[frame]
            frame_xyz.append([x,y,z])
    frame_xyz = np.array(frame_xyz)
    x_range = [np.min(frame_xyz[:,0]), np.max(frame_xyz[:,0])]
    y_range = [np.min(frame_xyz[:,1]), np.max(frame_xyz[:,1])]
    z_range = [np.min(frame_xyz[:,2]), np.max(frame_xyz[:,2])]
    all_xyz.append(frame_xyz)
all_xyz = np.array(all_xyz)
z_range = [np.min(all_xyz[:,:,2]), np.max(all_xyz[:,:,2])]
y_range = [np.min(all_xyz[:,:,1]), np.max(all_xyz[:,:,1])]
x_range = [np.min(all_xyz[:,:,0]), np.max(all_xyz[:,:,0])]

# Construct normalization colormap, normalized clormap
norm = matplotlib.colors.Normalize(vmin=z_range[0], vmax=z_range[1])
mymap = plt.get_cmap('viridis')
mymap = matplotlib.cm.ScalarMappable(norm=norm, cmap=mymap)
all_colors = mymap.to_rgba(all_xyz[:,:,2])

#new_map = plt.cm.viridis(all_xyz[:,:,2])


fig2 = plt.figure(2)
my_plot = plt.scatter(all_xyz[0,:,0], all_xyz[0,:,1], c=all_colors[0], cmap=mymap)
#my_plot = plt.scatter(all_xyz[0,:,0], all_xyz[0,:,1], color=new_map[0])
mymap.set_array([])
plt.colorbar(mymap)

fig2.savefig('temp.jpg')
plt.show()


fig1 = plt.figure(1)
my_plot = plt.scatter([], [],cmap=mymap, s=120)
plt.colorbar(mymap)

plt.xlim([x_range[0], x_range[1]])
plt.ylim([y_range[0], y_range[1]])
plt.xlabel("X (nm)")
plt.ylabel("Y (nm)")




scatter_animation = animation.FuncAnimation(fig1, _animate,
        frames=np.arange(traj.n_frames),
        fargs=(my_plot, all_xyz, all_colors, mymap))
scatter_animation.save("animation.mp4")
#
#


#fig, ax = plt.subplots(1,1, figsize=(8,6))
#figure =ax.scatter(all_xyz[:,0], all_xyz[:,1], c=all_xyz[:,2],
#        vmin = z_range[0], vmax=z_range[1], cmap='viridis', s=80)
#ax.set_xlabel("x-coordinate (nm)", fontsize=20)
#ax.set_ylabel("y-coordinate (nm)", fontsize=20)
#plt.colorbar(figure)
#fig.savefig("bot.svg",transparent=True)



