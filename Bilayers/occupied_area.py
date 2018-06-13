from __future__ import print_function
import mdtraj as mdtraj
import time
import bilayer_analysis_functions 
import itertools
from optparse import OptionParser
import numpy as np
from multiprocessing import Pool

def compute_occupied_profile_all(traj lipid_dict, bin_spacing=0.1, z_bins=None, centered=True):
    """ Compute void fraction  according to bins

    Parameters
    -----------
    traj : mdtraj trajectory
        Single trajectory frame
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
        traj.xyz[:,:,:] += center - com_z


    lipid_atoms = [atom.index for residue in traj.topology.residues 
                    if not residue.is_water for atom in residue.atoms]
    # Identify limits for bins, unless they've been provided
    if z_bins is None:
        bounds = (np.min(traj.xyz[:, lipid_atoms, 2]),
                np.max(traj.xyz[:, lipid_atoms,2]))
        n_bins = int(round((bounds[1] - bounds[0]) / bin_spacing))
        bin_width = (bounds[1] - bounds[0]) / n_bins
        z_bins = np.arange(bounds[0], bounds[1] + bin_spacing, bin_spacing)


        
    # Need to pass lists of parameters for the mapping function
    with Pool() as pool:
        occupied_profile = pool.starmap(_compute_occupied_profile_frame,
                            zip(itertools.repeat(traj), itertools.repeat(lipid_atoms),
                                itertools.repeat(z_bins), range(traj.n_frames)))
    return occupied_profile

def _compute_occupied_profile_frame(traj, lipid_atoms, z_bins, frame_i):
    """ Compute bilayer occupied area profile for a particular traj slice

    Parameters
    -----------
    traj : mdtraj trajectory
        Single trajectory frame
    lipid_atoms : list
        list of atom indices corresponding to lipids
    z_bins : np.array
        Z-coordinates for area calculation
    frame_i : int
        frame index to compute occupied area profile

    Returns
    -------
    f_occ : np.array
        occupied area profile for this frame 

    Notes
    -----
    Atoms are assumed to be vdW spheres with vDW radii

    """
    (x_length, y_length, z_length) = traj.unitcell_lengths[frame_i]
    occupied_area_profile = np.zeros_like(z_bins)
    total_area = x_length * y_length
    for i in lipid_atoms:
        atom_i = topol.atom(i)
        radius_i = atom_i.element.radius
        atom_i_center = traj.xyz[frame_i, i, :]

        # r_eff_sq is a numpy array of the different effective, squared radii
        # at the different z_bins
        r_eff_sq = ((radius_i**2) - ((z_bins - atom_i_center[2]) ** 2))

        # There may be elements in r_eff_sq that are negative becuase they are so 
        # far away from that z_bin, so add 0 instead
        occupied_area_profile += max(np.pi * (r_eff_sq), 0)

    f_occ_profile = occupied_area/total_area

    return frame_i, f_occ_profile


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


