import os
import pandas as pd
import numpy as np
import mdtraj
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import permeability as prm
import pdb

def main():
    #df = pd.read_pickle('perm_29sweeps_1.0nm.pkl')
    df = pd.read_pickle('perm_20sweeps_1.0nm.pkl')
    """
    df columns
    ----------
    'd_from_leaflet_i' : tracer distance from leaflet interface, float
    'd_from_local_i': tracer distance from local interface, float
    'diffusion' : diffusion coefficient, float
    'fcorr' : force autocorelation series, np array
    'forceout_index': Index for reading forceout files
    'free_energy' : free energy, float
    'grid_size' : width of XY grids, float
    'interface_bot': 2d matrix of interfaces, np.2darray
    'interface_top' : 2d matrix of interfaces, np.2darray
    'leaflet_interfaces' : list of leaflet interfaces, 2-tuple
    'local_interface' : coordinate of local interface
    'meanforce' : mean force , float
    'path',: path to tracer, list
    'permeability' : permeability, float
    'thickness',  : thickness of grid region
    'tracer_id' : tracer residue number, int
    'tracer_xyz' : tracer coordinates (from first frame), 3-tuple
    'xbin_centers' : centers of xbins, list
    'xedges': edges of xbins, list
    'ybin_centers': centers of ybins, list
    'yedges': edges of ybins, list
    
    Notes
    -----
    When looking at the distances from leaflet/local interface, 
    the convention is positive means closer to the bulk
    and negative means deeper within the bilayer
    
    
    Units
    -----
    Forces come from lammps, so they are kcal/mol/A
    Distances come from mdtraj analaysis, so thye are nm
    """
    
    leaflet_interfaces = (np.mean([el[0] for el in df.leaflet_interfaces.values]),
        np.mean([el[1] for el in df.leaflet_interfaces.values]))
    n_windows = len(list(set(df.loc[:,'forceout_index'].values)))
    n_sweeps = int(df.shape[0]/n_windows)
    
    
    ### ###
    ### Looking at local interface ###
    #######
    # compute bins
    bin_width = 0.2
    bounds = (np.min(df.d_from_local_i.values),
            np.max(df.d_from_local_i.values))
    n_bins = int(round((bounds[1] - bounds[0]) / bin_width))
    
    #d_from_leaflet_hist, edges = np.histogram(df.d_from_local_i.values, bins=10)
    #bin_width = edges[1] - edges[0]
    d_from_leaflet_hist, edges = np.histogram(df.d_from_local_i.values, bins=n_bins)
    bin_centers = edges[1:] - (bin_width / 2)
    
    d_from_local_i_p = pd.DataFrame(data=bin_centers, columns=['bin_center'])
    
    # Focus on the free energy first, I need to integrate from one end to the other
    dz = 2
    raw_mean_f = [[] for _ in range(len(bin_centers))]
    raw_forceout_index = [[] for _ in range(len(bin_centers))]
    # Start by grouping all the mean forces based on their bins
    for i, series in df.iterrows():
        bin_i = np.digitize(series.d_from_local_i, edges) - 1
        if bin_i >= len(bin_centers):
            bin_i = len(bin_centers)-1
        if bin_i < 0:
            bin_i = 0
        closest_i = np.argmin([np.abs(leaflet_i - series.local_interface) \
                for leaflet_i in series.leaflet_interfaces])
        if closest_i == 0:
            raw_mean_f[bin_i].append(series.meanforce)
        else:
            raw_mean_f[bin_i].append(-1 * series.meanforce)
        raw_forceout_index[bin_i].append(series.forceout_index)
    
    # Compute average of all mean forces found in that bin
    for bin_i, fe_series in enumerate(raw_mean_f):
        d_from_local_i_p.loc[bin_i, 'mean_force'] = np.mean(fe_series)
        d_from_local_i_p.loc[bin_i, 'mean_force_std'] = np.std(fe_series) / np.sqrt(len(fe_series))
        d_from_local_i_p.loc[bin_i, 'n_tracers'] = len(fe_series)
    
    
    # Now integrate to get the free enregy profile
    # Be mindful of the direction, integrating from bulk to the bilayer interior
    d_from_local_i_p.loc[:, 'free_energy'] = -np.cumsum(d_from_local_i_p.loc[::-1, 'mean_force']) * bin_width * 10
    d_from_local_i_p.loc[:, 'free_energy_std'] = np.sqrt(np.cumsum(\
            d_from_local_i_p.loc[::-1, 'mean_force_std']**2))

    
    
    raw_int_facf = [[] for _ in range(len(bin_centers))]
    # Doing diffusion things
    for i, series in df.iterrows():
        bin_i = np.digitize(series.d_from_local_i, edges) - 1
        if bin_i >= len(bin_centers):
            bin_i = len(bin_centers)-1
        if bin_i < 0:
            bin_i = 0
    
        average_fraction = 0.1
        time, FACF = series.fcorr[:,0], series.fcorr[:,1]
        intF = np.cumsum(FACF)*(time[1]-time[0])
        lastbit = int((1.0-average_fraction)*intF.shape[0])
        intFval = np.mean(intF[-lastbit:])
        raw_int_facf[bin_i].append(intFval)
    kb = 1.987e-3
    T = 305
    RT2 = (kb*T)**2
    RT2 *= 1e-4
    total_resistance = 0
    for bin_i, int_f_acf in enumerate(raw_int_facf):
        d_from_local_i_p.loc[bin_i, 'int_facf'] = np.mean(int_f_acf)
        d_from_local_i_p.loc[bin_i, 'diffusion'] = RT2/np.mean(int_f_acf)
        diff_vals = [RT2/val for val in int_f_acf]
        d_from_local_i_p.loc[bin_i, 'diffusion_std'] = np.std(diff_vals) / np.sqrt(len(diff_vals))
        d_from_local_i_p.loc[bin_i, 'resistance'] = np.exp(d_from_local_i_p.loc[bin_i, 'free_energy'] / (kb * T)) / d_from_local_i_p.loc[bin_i, 'diffusion']
        total_resistance += d_from_local_i_p.loc[bin_i, 'resistance'] * dz * 1e-8
    d_from_local_i_permeability = 1/(2*total_resistance)


    expdGerr = np.exp(d_from_local_i_p.loc[:, 'free_energy'] / (kb*T)) * \
            (d_from_local_i_p.loc[:,'free_energy_std'] /(kb*T))

    resist_err = d_from_local_i_p.loc[:,'resistance'] * \
            np.sqrt((expdGerr/np.exp(d_from_local_i_p.loc[:,'free_energy'] / (kb*T)))**2 + \
            (d_from_local_i_p.loc[:, 'diffusion_std']/d_from_local_i_p.loc[:,'diffusion'])**2) 
    R_err_global = np.sqrt(np.sum(resist_err**2) * dz) * 1e-8 # s/cm
    d_from_local_i_permeability_err = R_err_global * (d_from_local_i_permeability**2) / 2

            
            
    ####################################
    ### Looking at leaflet interface ###
    ####################################
    # compute bins
    bin_width = 0.2
    bounds = (np.min(df.d_from_leaflet_i.values),
            np.max(df.d_from_leaflet_i.values))
    n_bins = int(round((bounds[1] - bounds[0]) / bin_width))
    
    d_from_leaflet_hist, edges = np.histogram(df.d_from_leaflet_i.values, bins=n_bins)
    bin_centers = edges[1:] - (bin_width / 2)
    
    d_from_leaflet_i_p = pd.DataFrame(data=bin_centers, columns=['bin_center'])
    
    # Focus on the free energy first, I need to integrate from one end to the other
    dz = 2
    raw_mean_f = [[] for _ in range(len(bin_centers))]
    raw_forceout_index = [[] for _ in range(len(bin_centers))]
    # Start by grouping all the mean forces based on their bins
    for i, series in df.iterrows():
        bin_i = np.digitize(series.d_from_leaflet_i, edges) - 1
        if bin_i >= len(bin_centers):
            bin_i = len(bin_centers)-1
        if bin_i < 0:
            bin_i = 0
        closest_i = np.argmin([np.abs(leaflet_i - series.local_interface) \
                for leaflet_i in series.leaflet_interfaces])
        if closest_i == 0:
            raw_mean_f[bin_i].append(series.meanforce)
        else:
            raw_mean_f[bin_i].append(-1 * series.meanforce)
        raw_forceout_index[bin_i].append(series.forceout_index)
    
    # Compute average of all mean forces found in that bin
    for bin_i, fe_series in enumerate(raw_mean_f):
        d_from_leaflet_i_p.loc[bin_i, 'mean_force'] = np.mean(fe_series)
        d_from_leaflet_i_p.loc[bin_i, 'mean_force_std'] = np.std(fe_series) / np.sqrt(len(fe_series))
        d_from_leaflet_i_p.loc[bin_i, 'n_tracers'] = len(fe_series)
    
    
    # Now integrate to get the free enregy profile
    # Be mindful of the direction, integrating from bulk to the bilayer interior
    d_from_leaflet_i_p.loc[:, 'free_energy'] = -np.cumsum(d_from_leaflet_i_p.loc[::-1, 'mean_force']) * bin_width * 10
    # Get a standard deviation using propoagation of error from the mean force
    d_from_leaflet_i_p.loc[:, 'free_energy_std'] = np.sqrt(np.cumsum(\
            d_from_leaflet_i_p.loc[::-1, 'mean_force_std']**2))
    
    # Doing diffusion things
    raw_int_facf = [[] for _ in range(len(bin_centers))]
    for i, series in df.iterrows():
        bin_i = np.digitize(series.d_from_leaflet_i, edges) - 1
        if bin_i >= len(bin_centers):
            bin_i = len(bin_centers)-1
        if bin_i < 0:
            bin_i = 0
    
        average_fraction = 0.1
        time, FACF = series.fcorr[:,0], series.fcorr[:,1]
        intF = np.cumsum(FACF)*(time[1]-time[0])
        lastbit = int((1.0-average_fraction)*intF.shape[0])
        intFval = np.mean(intF[-lastbit:])
        raw_int_facf[bin_i].append(intFval)
    kb = 1.987e-3
    T = 305
    RT2 = (kb*T)**2
    RT2 *= 1e-4
    total_resistance = 0
    for bin_i, int_f_acf in enumerate(raw_int_facf):
        d_from_leaflet_i_p.loc[bin_i, 'int_facf'] = np.mean(int_f_acf)
        d_from_leaflet_i_p.loc[bin_i, 'diffusion'] = RT2/np.mean(int_f_acf)
        diff_vals = [RT2/val for val in int_f_acf]
        d_from_leaflet_i_p.loc[bin_i, 'diffusion_std'] = np.std(diff_vals) / np.sqrt(len(diff_vals))
        d_from_leaflet_i_p.loc[bin_i, 'resistance'] = np.exp(d_from_leaflet_i_p.loc[bin_i, 'free_energy'] / (kb * T)) / d_from_leaflet_i_p.loc[bin_i, 'diffusion']
        total_resistance += d_from_leaflet_i_p.loc[bin_i, 'resistance'] * dz * 1e-8

    d_from_leaflet_i_permeability = 1/(2*total_resistance)

    expdGerr = np.exp(d_from_leaflet_i_p.loc[:, 'free_energy'] / (kb*T)) * \
            (d_from_leaflet_i_p.loc[:,'free_energy_std'] /(kb*T))

    resist_err = d_from_leaflet_i_p.loc[:,'resistance'] * \
            np.sqrt((expdGerr/np.exp(d_from_leaflet_i_p.loc[:,'free_energy'] / (kb*T)))**2 + \
            (d_from_leaflet_i_p.loc[:, 'diffusion_std']/d_from_leaflet_i_p.loc[:,'diffusion'])**2) 
    R_err_global = np.sqrt(np.sum(resist_err**2) * dz) * 1e-8 # s/cm
    d_from_leaflet_i_permeability_err = R_err_global * (d_from_leaflet_i_permeability**2) / 2

    
    #################################
    ## Using absolute coordinates ##
    ################################
    old_profile = pd.DataFrame()
    old_raw_mean_f = [[] for _ in range(35)]
    for i, series in df.iterrows():
        old_bin_i = int(series.forceout_index)
        old_raw_mean_f[old_bin_i].append(series.meanforce)

    raw_mean_f = np.asarray(old_raw_mean_f)
    # Raw g profile is a free energy profile for each sweep
    raw_g_profile = np.zeros_like(raw_mean_f)
    for sweep,_ in enumerate(old_raw_mean_f[0]):
        raw_g_profile[:, sweep] = -np.cumsum(raw_mean_f[:, sweep]) * dz

    for bin_i, fe_series in enumerate(old_raw_mean_f):
        old_profile.loc[bin_i, 'mean_force'] = np.mean(fe_series)
        old_profile.loc[bin_i, 'mean_force_std'] = np.std(fe_series) / np.sqrt(len(fe_series))
        old_profile.loc[bin_i, 'n_tracers'] = len(fe_series)
    
    old_profile.loc[:,'free_energy'] = -np.cumsum(old_profile.loc[:, 'mean_force']) * dz
    old_profile.loc[:, 'free_energy_std'] = np.sqrt(np.cumsum(\
            old_profile.loc[:, 'mean_force_std']**2))

    old_raw_int_facf = [[] for _ in range(35)]
    for i, series in df.iterrows():
        average_fraction = 0.1
        time, FACF = series.fcorr[:,0], series.fcorr[:,1]
        intF = np.cumsum(FACF)*(time[1]-time[0])
        lastbit = int((1.0-average_fraction)*intF.shape[0])
        intFval = np.mean(intF[-lastbit:])
        old_bin_i = int(series.forceout_index)
        old_raw_int_facf[old_bin_i].append(intFval)
    
    total_resistance = 0
    for bin_i, int_f_acf in enumerate(old_raw_int_facf):
        old_profile.loc[bin_i, 'int_facf'] = np.mean(int_f_acf)
        old_profile.loc[bin_i, 'diffusion'] = RT2/np.mean(int_f_acf)
        diff_vals = [RT2/val for val in int_f_acf]
        old_profile.loc[bin_i, 'diffusion_std'] = np.std(diff_vals) / np.sqrt(len(diff_vals))
        old_profile.loc[bin_i, 'resistance'] = np.exp(old_profile.loc[bin_i, 'free_energy'] / (kb * T)) / old_profile.loc[bin_i, 'diffusion']
        total_resistance += old_profile.loc[bin_i, 'resistance'] * 2e-8

    absolute_permeability = 1/total_resistance
    expdGerr = np.exp(old_profile.loc[:, 'free_energy'] / (kb*T)) * \
            (old_profile.loc[:,'free_energy_std'] /(kb*T))


    resist_err = old_profile.loc[bin_i, 'resistance'] * \
            np.sqrt((expdGerr/np.exp(old_profile.loc[:,'free_energy'] / (kb*T)))**2 + \
            (old_profile.loc[:,'diffusion_std']/old_profile.loc[:,'diffusion'])**2) 
    R_err_global = np.sqrt(np.sum(resist_err**2) * dz) * 1e-8 # s/cm
    absolute_permeability_err = R_err_global * (absolute_permeability ** 2 )


    
    #######################################
    ## Symmetrizing absolute coordinaes ###
    #######################################
    old_g_sym_all = symmetrize_each(raw_g_profile, zero_boundary_condition=True)
    old_g_sym = np.mean(old_g_sym_all,axis=1)
    old_g_sym_err = np.std(old_g_sym_all, axis=1) / np.sqrt(n_sweeps)
    expdGerr = np.exp(old_g_sym / (kb*T)) * old_g_sym_err / (kb*T) 

    old_raw_int_facf_sym = symmetrize_each(np.asarray(old_raw_int_facf))
    diff_coeff_sym = RT2/np.mean(old_raw_int_facf_sym, axis=1)
    diff_coeff_sym_err = RT2*np.std(old_raw_int_facf_sym, axis=1) / \
            (np.mean(old_raw_int_facf_sym, axis=1)**2) / np.sqrt(n_sweeps)
    
    resistance = np.exp(old_g_sym / (kb*T))/diff_coeff_sym  
    absolute_permeability_sym = 1/(np.sum(resistance) * dz * 1e-8)


    resist_err = resistance * np.sqrt((expdGerr/np.exp(old_g_sym / (kb*T)))**2+(diff_coeff_sym_err/diff_coeff_sym)**2) 
    R_err_global = np.sqrt(np.sum(resist_err**2) * dz) * 1e-8 # s/cm
    absolute_permeability_sym_err  = R_err_global * (absolute_permeability_sym**2)
    
    print("{0},{1},{2},{3},{4},{5},{6},{7}".format(
        d_from_local_i_permeability, d_from_local_i_permeability_err,
        d_from_leaflet_i_permeability, d_from_leaflet_i_permeability_err,
        absolute_permeability, absolute_permeability_err,
        absolute_permeability_sym, absolute_permeability_sym_err))
    
    
    
    
    
    # mean forces
    #fig = plt.figure(1)
    #x_vals = np.arange(0, 0.2*old_profile.shape[0], 0.2)
    #plt.errorbar(x_vals, old_profile.mean_force.values, 
    #        yerr=old_profile.mean_force_std.values,
    #        label='absolute coordinates')
    #for interface in leaflet_interfaces:
    #    plt.axvline(x=interface, color='r')
    #plt.ylabel("Mean force (kcal/$\AA$ mol)", fontsize=20)
    #plt.xlabel("Distance (nm)", fontsize=20)
    #plt.legend()
    #plt.savefig("absolute_xyz_meanf.jpg")
    #plt.close()
    #
    #fig = plt.figure(1)
    #plt.errorbar(profile.bin_center.values, profile.mean_force.values, 
    #        yerr = profile.mean_force_std.values, 
    #        label="Distance from local interface")
    #plt.ylabel("Mean force (kcal/$\AA$ mol)", fontsize=20)
    #plt.xlabel("Distance (nm)", fontsize=20)
    #np.savetxt('dfromleafleti_meanf.dat', np.column_stack((profile.bin_center.values,
    #    profile.mean_force.values, profile.mean_force_std.values)))
    #plt.legend()
    #plt.savefig("dfromi_xyz_meanf.jpg")
    #plt.close()
    #
    ## free energy
    #fig = plt.figure(1)
    #plt.plot(x_vals, old_profile.free_energy.values, label='absolute coordinates')
    #for interface in leaflet_interfaces:
    #    plt.axvline(x=interface, color='r')
    #plt.ylabel("Free energy (kcal/mol)", fontsize=20)
    #plt.xlabel("Distance (nm)", fontsize=20)
    #plt.legend()
    #plt.savefig("absolute_xyz_fe.jpg")
    #plt.close()
    #
    #fig = plt.figure(1)
    #plt.plot(profile.bin_center.values, profile.free_energy.values, 
    #        label='Distance from local interface')
    #plt.ylabel("Free energy (kcal/mol)", fontsize=20)
    #plt.xlabel("Distance (nm)", fontsize=20)
    #np.savetxt('dfromleafleti_fe.dat',np.column_stack((profile.bin_center.values, 
    #    profile.free_energy.values)))
    #plt.legend()
    #plt.savefig("dfromi_xyz_fe.jpg")
    #plt.close()
    #
    #
    #
    ## Diffusion
    #fig = plt.figure(1)
    #plt.errorbar(x_vals, old_profile.diffusion.values, 
    #        yerr=  old_profile.diffusion_std.values,
    #        label='absolute coordinates')
    #for interface in leaflet_interfaces:
    #    plt.axvline(x=interface, color='r')
    #
    #plt.yscale("log")
    #plt.ylabel("Diffusion (cm$^2$/sec)", fontsize=20)
    #plt.xlabel("Distance (nm)", fontsize=20)
    #plt.legend()
    #plt.savefig('absolute_xyz_diff.jpg')
    #plt.close()
    #
    #fig = plt.figure(1)
    #plt.errorbar(profile.bin_center.values, profile.diffusion.values, 
    #        yerr = profile.diffusion_std.values,
    #        label='Distance from local interface')
    #plt.yscale("log")
    #plt.ylabel("Diffusion (cm$^2$/sec)", fontsize=20)
    #plt.xlabel("Distance (nm)", fontsize=20)
    #plt.legend()
    #plt.savefig('dfromi_xyz_diff.jpg')
    #np.savetxt('dfromleafleti_diff.dat', np.column_stack((profile.bin_center.values,
    #    profile.diffusion.values, profile.diffusion_std.values)))
    #plt.close()
    #
    #
    #
    #pdb.set_trace()

def symmetrize_each(data, zero_boundary_condition=False):
    """Symmetrize a profile
    
    Params
    ------
    data : np.ndarray, shape=(n,n_sweeps)
        Data to be symmetrized
    zero_boundary_condition : bool, default=False
        If True, shift the right half of the curve before symmetrizing

    Returns
    -------
    dataSym : np.ndarray, shape=(n,)
        symmetrized data

    This function symmetrizes a 1D array. It also provides an error estimate
    for each value, taken as the standard error between the "left" and "right"
    values. The zero_boundary_condition shifts the "right" half of the curve 
    such that the final value goes to 0. This should be used if the data is 
    expected to approach zero, e.g., in the case of pulling a water molecule 
    through one phase into bulk water.
    """
    n_sweeps = data.shape[1]
    n_windows = data.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dataSym = np.zeros_like(data)
    for s in range(n_sweeps):
        for i, sym_val in enumerate(dataSym[:n_win_half,s]):
            val = 0.5 * (data[i,s] + data[-(i+1),s])
            dataSym[i,s] = val
            dataSym[-(i+1),s] = val
        if zero_boundary_condition:
            dataSym[:,s] -= dataSym[0,s] 
    return dataSym

def symmetrize(data, zero_boundary_condition=False):
    """Symmetrize a profile
    
    Params
    ------
    data : np.ndarray, shape=(n,)
        Data to be symmetrized
    zero_boundary_condition : bool, default=False
        If True, shift the right half of the curve before symmetrizing

    Returns
    -------
    dataSym : np.ndarray, shape=(n,)
        symmetrized data
    dataSym_err : np.ndarray, shape=(n,)
        error estimate in symmetrized data

    This function symmetrizes a 1D array. It also provides an error estimate
    for each value, taken as the standard error between the "left" and "right"
    values. The zero_boundary_condition shifts the "right" half of the curve 
    such that the final value goes to 0. This should be used if the data is 
    expected to approach zero, e.g., in the case of pulling a water molecule 
    through one phase into bulk water.
    """
    n_windows = data.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dataSym = np.zeros_like(data)
    dataSym_err = np.zeros_like(data)
    shift = {True: data[-1], False: 0.0}
    for i, sym_val in enumerate(dataSym[:n_win_half]):
        val = 0.5 * (data[i] + data[-(i+1)])
        err = np.std([data[i], data[-(i+1)] - shift[zero_boundary_condition]]) / np.sqrt(2)
        dataSym[i], dataSym_err[i] = val, err
        dataSym[-(i+1)], dataSym_err[-(i+1)] = val, err        
    return dataSym, dataSym_err

if __name__ == "__main__":
    main()
