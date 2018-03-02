import numpy as np
import itertools
import pdb
import mdtraj
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

""" Compute density profiles for lipids and water
also compute density profiles for smaller XY grids"""


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

    return np.array(density_profile), xbin_centers, ybin_centers


def _find_interface(water_p, bins, rho_interface=984,frame=0,reverse=False):
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
    midpoint = int(np.shape(water_p)[1]/2)
    # Find closest indices to rho_interface
    s = np.argsort(np.abs(water_p[0, : midpoint] - rho_interface))

    # Use linear interpolation to find more precise value of interface
    z_interface_bot = np.interp(rho_interface,
            [water_p[0,s[0]], water_p[0,s[1]], water_p[0,s[2]]], 
            [bins[s[0]], bins[s[1]], bins[s[1]]])


    # Find closest indices to rho_interface
    s = np.argsort(np.abs(water_p[0, midpoint : ] - rho_interface))

    # Use linear interpolation to find more precise value of interface
    z_interface_top = np.interp(rho_interface,
            [water_p[0,midpoint+s[0]], water_p[0, midpoint+s[1]], water_p[0, midpoint+s[2]]], 
            [bins[midpoint+s[0]], bins[midpoint+s[1]], bins[midpoint+ s[1]]])


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

if __name__ == "__main__":
    grofile = 'centered.gro'
    #traj = mdtraj.load(xtcfile, top=grofile)
    traj = mdtraj.load(grofile)


    # Water density profile over a region
    water_indices = traj.topology.select('water') 
    water_p, bins = calc_density_profile(traj, traj.topology, water_indices,
            l_x=traj.unitcell_lengths[0,0],
            l_y=traj.unitcell_lengths[0,1])
    fig2=plt.figure(2)
    plt.plot(bins, water_p[0])
    plt.savefig("water_profile.svg",transparent=True)


    # Identification of top and bottom interfaces
    rho_water = 984 # SPC water
    rho_interface = rho_water / np.e # Definition of water interface
    z_interface_bot, z_interface_top = _find_interface(water_p, bins, rho_interface=rho_interface)
    interfaces = [z_interface_bot, z_interface_top]


    # Generate 2D histogram of density 
    all_indices = [a.index for a in traj.topology.atoms if a.residue.is_water]
    thickness = 0.5
    for i, z_interface in enumerate(interfaces):
        if i == 0:
            name = "botwater"
        else:
            name = "topwater"

        # Given location of z_interface, find the atoms around it
        atoms_z = _find_atoms_around(traj, all_indices, thickness=thickness,
                z=z_interface)
        vmd_string = " ".join(["{}".format(thing) for thing in atoms_z])

        # Given that subset of atoms, then use the 2d histogram 
        density_surface, xbin_centers, ybin_centers = calc_density_surface(traj,
                atoms_z, grid_size=0.5, thickness=thickness)
        fig1 = plt.figure(1)

        # Yes we need to plot the transpose, otherwise the spatial X coords
        # get plotted on the Y axis and the Y coords get plotted on the X axis
        plt.imshow(density_surface[0,:,:].T, cmap='viridis', origin='lower')
        plt.colorbar()


        xtick_vals, step = np.linspace(0, density_surface.shape[1], num=5, 
                dtype=int, retstep=True, endpoint=False)
        xtick_labels = [np.round(x,2) for x in xbin_centers][::int(step)]
        plt.xticks(xtick_vals,  xtick_labels)

        ytick_vals, step = np.linspace(0, density_surface.shape[2], num=5,
                dtype=int, retstep=True, endpoint=False)
        ytick_labels = [np.round(y,2) for y in ybin_centers][::int(step)]
        plt.yticks(ytick_vals,  ytick_labels)
        #plt.xticks(np.arange(0, density_surface.shape[1], step=5), 
        #        [np.round(x,2) for x in xbin_centers][::5])
        #plt.yticks(np.arange(density_surface.shape[2], step=5), 
        #        [np.round(y,2) for y in ybin_centers][::5])
        plt.savefig('{}.svg'.format(name), transparent=True)
        plt.close()

    # Using the 2D hist bins, find the z-interface in each grid

    xbin_width = xbin_centers[1] - xbin_centers[0]
    ybin_width = ybin_centers[1] - ybin_centers[0]

    interface_bot_surface = np.zeros_like(density_surface)
    interface_top_surface = np.zeros_like(density_surface)
    for x, y in itertools.product(xbin_centers, ybin_centers):
        atoms_xy = _find_atoms_within(traj, x=x, y=y, atom_indices=water_indices,
                xbin_width=xbin_width, ybin_width=ybin_width)
        profile_xy, bins = calc_density_profile(traj, traj.topology, atoms_xy,
                l_x=xbin_width, l_y=ybin_width)
        z_interface_bot, z_interface_top = _find_interface(profile_xy, bins,
                rho_interface=rho_interface)
        interface_bot_surface[ 0,int(np.floor(x/xbin_width)), 
                int(np.floor(y/ybin_width))] = z_interface_bot
        interface_top_surface[ 0,int(np.floor(x/xbin_width)), 
                int(np.floor(y/ybin_width))] = z_interface_top


    fig3 = plt.figure(3)
    plt.imshow(interface_bot_surface[0, :, :].T, cmap='viridis', origin='lower')
    plt.colorbar()

    xtick_vals, step = np.linspace(0, interface_bot_surface.shape[1], num=5, 
            dtype=int, retstep=True, endpoint=False)
    xtick_labels = [np.round(x,2) for x in xbin_centers][::int(step)]
    plt.xticks(xtick_vals,  xtick_labels)

    ytick_vals, step = np.linspace(0, interface_bot_surface.shape[2], num=5,
            dtype=int, retstep=True, endpoint=False)
    ytick_labels = [np.round(y,2) for y in ybin_centers][::int(step)]
    plt.yticks(ytick_vals,  ytick_labels)

    plt.savefig('interface_bot_surface.svg', transparent=True)
    plt.close()

    fig4 = plt.figure(4)
    plt.imshow(interface_top_surface[0, :, :].T, cmap='viridis', origin='lower')
    plt.colorbar()

    xtick_vals, step = np.linspace(0, interface_top_surface.shape[1], num=5, 
            dtype=int, retstep=True, endpoint=False)
    xtick_labels = [np.round(x,2) for x in xbin_centers][::int(step)]
    plt.xticks(xtick_vals,  xtick_labels)

    ytick_vals, step = np.linspace(0, interface_top_surface.shape[2], num=5,
            dtype=int, retstep=True, endpoint=False)
    ytick_labels = [np.round(y,2) for y in ybin_centers][::int(step)]
    plt.yticks(ytick_vals,  ytick_labels)

    plt.savefig('interface_top_surface.svg', transparent=True)
    plt.close()

