from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import mdtraj as mdtraj
import time
import bilayer_analysis_functions 
import itertools
import numpy as np
from multiprocessing import Pool

def compute_occupied_profile_all(traj, bin_spacing=0.1, z_bins=None, centered=True,
                                plot=True):
    """ Compute void fraction  according to bins

    Parameters
    -----------
    traj : mdtraj trajectory
        Single trajectory frame
    bin_spacing : float, opt
        Distance between slices [nm]
    z_bins : np.array, default None
        Array of z bins we are evaluating the occupied area at
    centered : bool, opt,  default True
        If True, re-center the bilayer
    plot : bool ,opt, default True
        If True, save the occupied area profile and plot

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
        occupied_area_profile = pool.starmap(_compute_occupied_profile_frame,
                            zip(itertools.repeat(traj), itertools.repeat(lipid_atoms),
                                itertools.repeat(z_bins), range(traj.n_frames)))
    occupied_area_profile = np.array(occupied_area_profile)

    if plot:
        mean_profile = np.mean(occupied_area_profile, axis=0)
        std_profile = np.std(occupied_area_profile, axis=0)
    
        np.savetxt('occupied_area_profile.dat', np.column_stack((z_bins, 
                                                mean_profile, std_profile)))
        fig, ax = plt.subplots(1,1)
        ax.plot(z_bins, mean_profile)
        ax.fill_between(z_bins, mean_profile - std_profile, 
                        mean_profile + std_profile,
                        alpha=0.4)
        ax.set_ylabel("Occupied Area Fraction")
        ax.set_xlabel("Depth (nm)")
        ax.set_ylim([0,1])
        fig.savefig('occupied_area_profile.png')
        plt.close(fig)

    return z_bins, occupied_area_profile

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
        atom_i = traj.topology.atom(i)
        if "H" not in atom_i.element.symbol:
            radius_i = atom_i.element.radius
            atom_i_center = traj.xyz[frame_i, i, :]

            # r_eff_sq is a numpy array of the different effective, squared radii
            # at the different z_bins
            r_eff_sq = ((radius_i**2) - ((z_bins - atom_i_center[2]) ** 2))

            # There may be elements in r_eff_sq that are negative becuase they are so 
            # far away from that z_bin, so add 0 instead
            for i, val in enumerate(r_eff_sq):
                occupied_area_profile[i] += max(np.pi * val, 0)

    f_occ_profile = occupied_area_profile/total_area
    # Don't let occupied fractions exceed 1
    f_occ_profile = np.array([min(f,1) for f in f_occ_profile])

    return  f_occ_profile


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
    traj = mdtraj.load('npt_80-100ns.xtc', top='npt.gro')


    start = time.time()
    z_bins, occupied_area_profile = compute_occupied_profile_all(traj)
    end = time.time()
    #print('Occ profile took: {}'.format(end-start))



