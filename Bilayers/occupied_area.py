from __future__ import print_function
import mdtraj as mdtraj
import time
import bilayer_analysis_functions 
import itertools
from optparse import OptionParser
import numpy as np
from multiprocessing import Pool

def compute_occupied_profile_all(traj, topol, lipid_dict, bin_spacing =0.1,centered=True):
    """ Compute void fraction  according to bins

    Parameters
    -----------
    traj : mdtraj trajectory
        Single trajectory frame
    topol : mdtraj topology
    lipid_dict : dict
        Mapping residue indices to associated atom indices
    bin_spacing : float
        Distance between slices [nm]

    Returns
    -------
    void_fraction_profile : ndarray, shape = (n_bins, 1)
        Void fracion at various bins

    Notes
    -----
    Atoms are assumed to be vdW spheres with vDW radii
    """

    # Place the center of mass of the bilayer at 5 (arbitrary)
    if centered:
        com_z = _compute_com(traj)
        center = 5
        traj.xyz[0,:,:] += center - com_z


    # Identify limits for bins
    z_min = np.min( traj.atom_slice([atom for atom in itertools.chain.from_iterable(lipid_dict.values())]).xyz[:,:,2])
    z_max = np.max( traj.atom_slice([atom for atom in itertools.chain.from_iterable(lipid_dict.values())]).xyz[:,:,2])
    z_min -= 0.5
    z_max += 0.5
    z_bins = np.arange(z_min, z_max, bin_spacing)

    z_profile = []
    f_occ_profile = []
    # Need to pass lists of parameters for the mapping function
    print("starting pooling")
    with Pool() as pool:
        occupied_profile = np.asarray(pool.starmap(_compute_occupied_profile_slice, 
                zip(itertools.repeat(traj), itertools.repeat(topol), itertools.repeat(lipid_dict), z_bins)))
    print("done pooling")
    return occupied_profile

def _compute_occupied_profile_slice(traj, topol, lipid_dict, z_bin):
    """ Compute bilayer occupied area for a particular lateral slice

    Parameters
    -----------
    traj : mdtraj trajectory
        Single trajectory frame
    topol : mdtraj topology
    lipid_dict : dict
        Mapping residue indices to associated atom indices
    z_bin : float
        Z-coordinate for area calculation 

    Returns
    -------
    f_occ : float
        occupied area at particular z-coordinate

    Notes
    -----
    Atoms are assumed to be vdW spheres with vDW radii

    """
    (x_length, y_length, z_length) = traj.unitcell_lengths[0]
    atoms_effective = []
    occupied_area = 0
    total_area = x_length * y_length
    for i in itertools.chain.from_iterable(lipid_dict.values()):
        atom_i = topol.atom(i)
        if not atom_i.residue.is_water:
            radius_i = atom_i.element.radius
            atom_i_center = traj.atom_slice([i]).xyz[0, 0, :]
            r_eff_sq = ((radius_i**2) - ((z_bin - atom_i_center[2]) ** 2))
            if r_eff_sq >= 0.00:
                occupied_area += np.pi * (r_eff_sq)

            # Assuming vdW spheres don't overlap, evaluate area of spheres 
            # Compared to box's xy area

    f_occ = occupied_area/total_area

    return z_bin,f_occ


def _compute_com( traj):
    """ Compute center of mass

    Parameters
    -----------
    traj : mdtraj trajectory
        Single trajectory frame
    
    Returns
    -------
   com_z: float 
    """
    numerator = sum(atom.element.mass*traj.xyz[0,atom.index,:][2] for atom in traj.top.atoms)
    totalmass = sum(atom.element.mass for atom in traj.top.atoms)
    com_z = numerator/totalmass 
    return com_z


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-f', action="store", type="string", default = 'nopbc.xtc', dest = 'trajfile')
    parser.add_option('-c', action="store", type="string", default = 'nopbc.gro', dest = 'grofile')
    parser.add_option('-o', action="store", type="string", default = 'occupied_area.txt', dest = 'outfile')
    (options, args) = parser.parse_args()

    outfile = open(options.outfile, 'w')

    print('Loading trajectory <{}>...'.format(options.trajfile))
    print('Loading topology <{}>...'.format(options.grofile))
    traj = mdtraj.load(trajfile, top=options.grofile)
    topol = traj.topology

    # Compute system information
    print('Gathering system information <{}>...'.format(options.grofile))
    lipid_dict, headgroup_dict = bilayer_analysis_functions.get_lipids(topol)
    lipid_tails = bilayer_analysis_functions.get_lipid_tails(topol, lipid_dict)

    start = time.time()
    occupied_area_profile = compute_occupied_profile_all(traj[0], topol, lipid_dict, bin_spacing = 0.1, centered=True)
    end = time.time()
    print("{:30s}: {}".format(options.grofile[:-4], end-start))
    np.savetxt(outfile, occupied_area_profile)


