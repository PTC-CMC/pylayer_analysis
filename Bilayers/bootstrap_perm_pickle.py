import os
import itertools
import pandas as pd
import numpy as np
import mdtraj
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import permeability as prm
import pdb

def main(bs_sample_size=None, n_bs=1):
    df = pd.read_pickle('perm_pickle_1.0nm.pkl')
    #df = pd.read_pickle('perm_pickle_sweep0rh.pkl')
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
    ----
    Forces come from lammps, so they are kcal/mol/A
    Distances come from mdtraj analaysis, so thye are nm
    """
    
    leaflet_interfaces = (np.mean([el[0] for el in df.leaflet_interfaces.values]),
        np.mean([el[1] for el in df.leaflet_interfaces.values]))
            
            
    # compute bins
    bin_width = 0.2
    #bs_sample_size = 50 # In a given sample, how many sweeps/tracers do we look at, or how many elements are in the resample
    #n_bs = 1 # Kind of like number of time origins, how many times we resample
        
    bounds = (np.min(df.d_from_leaflet_i.values),
            np.max(df.d_from_leaflet_i.values))
    n_bins = int(round((bounds[1] - bounds[0]) / bin_width))
    
    ##################
    ## Looking at distance from local interface ##
    #################
    #d_from_leaflet_hist, edges = np.histogram(df.d_from_leaflet_i.values, bins=10)
    #bin_width = edges[1] - edges[0]
    d_from_local_hist, edges = np.histogram(df.d_from_local_i.values, bins=n_bins)
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

    # Since each window might have different number of samples, identify 
    # bs indices one at a time
    bs_indices = []
    for fe_series in raw_mean_f:
        window_bs_indices = _select_bootstrap_indices(len(fe_series),
                n_bs, bs_sample_size)
        bs_indices.append(window_bs_indices)
        
    bs_mean_forces = np.zeros((len(bin_centers), n_bs))
    # Compute average of all mean forces found in that bin
    for bin_i, fe_series in enumerate(raw_mean_f):
        for bs_num, bs_series_indices in enumerate(bs_indices[bin_i]):
           bs_series = [fe_series[int(i)] for i in bs_series_indices] 
           bs_mean_forces[bin_i, bs_num] = np.mean(bs_series)


        d_from_local_i_p.loc[bin_i, 'n_tracers'] = len(fe_series)
        d_from_local_i_p.loc[bin_i, 'mean_force'] = np.mean(bs_mean_forces[bin_i,:])
        d_from_local_i_p.loc[bin_i, 'mean_force_std'] = np.std(bs_mean_forces[bin_i,:])

        
                
    # Now integrate to get the free enregy d_from_local_i_p
    # Be mindful of the direction, integrating from bulk to the bilayer interior
    
    # Compute free energy profile for each bootstrap
    bs_g_profile = np.zeros((len(bin_centers), n_bs))
    for bs_num in range(n_bs):
        bs_g_profile[::-1, bs_num] = -np.cumsum(bs_mean_forces[::-1, bs_num]) * bin_width * 10
    for window in range(len(bin_centers)):
        d_from_local_i_p.loc[window, 'g_mean'] = np.mean(bs_g_profile[window,:])
        d_from_local_i_p.loc[window, 'g_err'] = np.std(bs_g_profile[window, :])
    
    
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

    bs_int_facf = np.zeros((len(bin_centers), n_bs))
    for bin_i, int_f_acf in enumerate(raw_int_facf):
        bs_sample = []
        for bs_i, bs_series_indices in enumerate(bs_indices[bin_i]):
           bs_series = [int_f_acf[int(i)] for i in bs_series_indices] 
           bs_sample.append(np.mean(bs_series))
           bs_int_facf[bin_i, bs_i] = np.mean(bs_series)
    
    
    for window in range(len(bin_centers)):
        # Estimates for diffusion mean can be taken from the bootstrapped
        # Force ACF integrals
       d_from_local_i_p.loc[window, 'd_mean'] = RT2/np.mean(bs_int_facf[window,:])
       d_from_local_i_p.loc[window, 'd_err'] = RT2/np.std(bs_int_facf[window,:])
       d_from_local_i_p.loc[window, 'r'] = np.exp(d_from_local_i_p.loc[window, 'g_mean'] / (kb * T)) / d_from_local_i_p.loc[window, 'd_mean']
       #total_resistance += d_from_local_i_p.loc[window, 'r'] * 2e-8
    #d_from_local_i_permeability = 1/(2*total_resistance)

    # Estimating permeability means calculating permeability for each 
    # bootstrapped sample, and then performing averaging/std
    bs_r_profile = np.zeros((len(bin_centers), n_bs))
    bs_d_profile = np.zeros((len(bin_centers), n_bs))
    bs_p = np.zeros(n_bs)
    for bs_i in range(n_bs):
        for window in range(len(bin_centers)):
           bs_d_profile[window, bs_i] = RT2/bs_int_facf[window,bs_i]
           bs_r_profile[window, bs_i] = np.exp(bs_g_profile[window, bs_i]/(kb*T)) / bs_d_profile[window,bs_i]
        bs_p[bs_i] = 1/(2*np.sum(bs_r_profile[:, bs_i]) * dz * 1e-8)

    d_from_local_i_permeability = np.mean(bs_p)
    d_from_local_i_permeability_err = np.std(bs_p)


       
    ###############
    ## Looking at distance from leaflet interface ##
    ###############

    bounds = (np.min(df.d_from_leaflet_i.values),
            np.max(df.d_from_leaflet_i.values))
    n_bins = int(round((bounds[1] - bounds[0]) / bin_width))
    
    #d_from_leaflet_hist, edges = np.histogram(df.d_from_leaflet_i.values, bins=10)
    #bin_width = edges[1] - edges[0]
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

    # Since each window might have different number of samples, identify 
    # bs indices one at a time
    bs_indices = []
    for fe_series in raw_mean_f:
        window_bs_indices = _select_bootstrap_indices(len(fe_series),
                n_bs, bs_sample_size)
        bs_indices.append(window_bs_indices)
        
    bs_mean_forces = np.zeros((len(bin_centers), n_bs))
    # Compute average of all mean forces found in that bin
    for bin_i, fe_series in enumerate(raw_mean_f):
        for bs_num, bs_series_indices in enumerate(bs_indices[bin_i]):
           bs_series = [fe_series[int(i)] for i in bs_series_indices] 
           bs_mean_forces[bin_i, bs_num] = np.mean(bs_series)


        d_from_leaflet_i_p.loc[bin_i, 'n_tracers'] = len(fe_series)
        d_from_leaflet_i_p.loc[bin_i, 'mean_force'] = np.mean(bs_mean_forces[bin_i,:])
        d_from_leaflet_i_p.loc[bin_i, 'mean_force_std'] = np.std(bs_mean_forces[bin_i,:])
        
                
    # Now integrate to get the free enregy d_from_leaflet_i_p
    # Be mindful of the direction, integrating from bulk to the bilayer interior
    
    # Compute free energy profile for each bootstrap
    bs_g_profile = np.zeros((len(bin_centers), n_bs))
    for bs_num in range(n_bs):
        bs_g_profile[::-1, bs_num] = -np.cumsum(bs_mean_forces[::-1, bs_num]) * bin_width * 10
    for window in range(len(bin_centers)):
        d_from_leaflet_i_p.loc[window, 'g_mean'] = np.mean(bs_g_profile[window,:])
        d_from_leaflet_i_p.loc[window, 'g_err'] = np.std(bs_g_profile[window, :])
    
    
    raw_int_facf = [[] for _ in range(len(bin_centers))]
    # Doing diffusion things
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

    bs_int_facf = np.zeros((len(bin_centers), n_bs))
    for bin_i, int_f_acf in enumerate(raw_int_facf):
        bs_sample = []
        for bs_i, bs_series_indices in enumerate(bs_indices[bin_i]):
           bs_series = [int_f_acf[int(i)] for i in bs_series_indices] 
           bs_sample.append(np.mean(bs_series))
           bs_int_facf[bin_i, bs_i] = np.mean(bs_series)
    
    for window in range(len(bin_centers)):
        # Diffusion estimates from the bootstrapped force ACF integrals
       d_from_leaflet_i_p.loc[window, 'd_mean'] = RT2/np.mean(bs_int_facf[window,:])
       d_from_leaflet_i_p.loc[window, 'd_err'] = RT2/np.std(bs_int_facf[window,:])
       d_from_leaflet_i_p.loc[window, 'r'] = np.exp(d_from_leaflet_i_p.loc[window, 'g_mean'] / (kb * T)) / d_from_leaflet_i_p.loc[window, 'd_mean']
       #total_resistance += d_from_leaflet_i_p.loc[window, 'r'] * 2e-8
    #d_from_leaflet_i_permeability = 1/(2*total_resistance)

    # Estimating permeability means calculating permeability for each 
    # bootstrapped sample, and then performing averaging/std
    bs_r_profile = np.zeros((len(bin_centers), n_bs))
    bs_d_profile = np.zeros((len(bin_centers), n_bs))
    bs_p = np.zeros(n_bs)
    for bs_i in range(n_bs):
        for window in range(len(bin_centers)):
           bs_d_profile[window, bs_i] = RT2/bs_int_facf[window,bs_i]
           bs_r_profile[window, bs_i] = np.exp(bs_g_profile[window, bs_i]/(kb*T)) / bs_d_profile[window,bs_i]
        bs_p[bs_i] = 1/(2*np.sum(bs_r_profile[:, bs_i]) * dz * 1e-8)

    d_from_leaflet_i_permeability = np.mean(bs_p)
    d_from_leaflet_i_permeability_err = np.std(bs_p)



        
    ###########
    ### Absolute coordinates ###
    ##############
    absolute_p = pd.DataFrame()
    n_forceouts = len(set(df.loc[:,'forceout_index'].values))
    
    # Focus on the free energy first, I need to integrate from one end to the other
    old_raw_mean_f = [[] for _ in range(n_forceouts)]
    
    # Start by grouping all the mean forces based on their bins
    for i, series in df.iterrows():
        old_bin_i = int(series.forceout_index)
        old_raw_mean_f[old_bin_i].append(series.meanforce)

    # Since each window might have different number of samples, identify 
    # bs indices one at a time
    old_bs_indices = [] 
    for fe_series in old_raw_mean_f:
        window_bs_indices = _select_bootstrap_indices(len(fe_series),
                n_bs, bs_sample_size)
        old_bs_indices.append(window_bs_indices)
    
    bs_mean_forces = np.zeros((n_forceouts, n_bs))
    # Compute average of all mean forces found in that bin
    for bin_i, fe_series in enumerate(old_raw_mean_f):
        for bs_num, bs_series_indices in enumerate(old_bs_indices[bin_i]):
           bs_series = [fe_series[int(i)] for i in bs_series_indices] 
           bs_mean_forces[bin_i, bs_num] = np.mean(bs_series)


        absolute_p.loc[bin_i, 'n_tracers'] = len(fe_series)
        absolute_p.loc[bin_i, 'mean_force'] = np.mean(bs_mean_forces[bin_i,:])
        absolute_p.loc[bin_i, 'mean_force_std'] = np.std(bs_mean_forces[bin_i,:])
    # Now integrate to get the free enregy d_from_leaflet_i_p
    # Be mindful of the direction, integrating from bulk to the bilayer interior
    bs_g_profile = np.zeros((n_forceouts, n_bs))
    for bs_num in range(n_bs):
        bs_g_profile[:, bs_num] = -np.cumsum(bs_mean_forces[:, bs_num]) * bin_width * 10
    for window in range(n_forceouts):
        absolute_p.loc[window, 'g_mean'] = np.mean(bs_g_profile[window,:])
        absolute_p.loc[window, 'g_err'] = np.std(bs_g_profile[window, :])

    
    
    
    old_raw_int_facf = [[] for _ in range(n_forceouts)]

    # Doing diffusion things
    for i, series in df.iterrows():
        average_fraction = 0.1
        time, FACF = series.fcorr[:,0], series.fcorr[:,1]
        intF = np.cumsum(FACF)*(time[1]-time[0])
        lastbit = int((1.0-average_fraction)*intF.shape[0])
        intFval = np.mean(intF[-lastbit:])
        old_bin_i = int(series.forceout_index)
        old_raw_int_facf[old_bin_i].append(intFval)
    kb = 1.987e-3
    T = 305
    RT2 = (kb*T)**2
    RT2 *= 1e-4
    total_resistance = 0

    bs_int_facf = np.zeros((n_forceouts, n_bs))
    for bin_i, int_f_acf in enumerate(old_raw_int_facf):
        for bs_i, bs_series_indices in enumerate(old_bs_indices[bin_i]):
           bs_series = [int_f_acf[int(i)] for i in bs_series_indices] 
           bs_int_facf[bin_i, bs_i] = np.mean(bs_series)
    
    for window in range(n_forceouts):
       absolute_p.loc[window, 'd_mean'] = RT2/np.mean(bs_int_facf[window,:])
       absolute_p.loc[window, 'd_err'] = RT2/np.std(bs_int_facf[window,:])
       absolute_p.loc[window, 'r'] = np.exp(absolute_p.loc[window, 'g_mean'] / (kb * T)) / absolute_p.loc[window, 'd_mean']
       #total_resistance += absolute_p.loc[window, 'r'] * 2e-8
    #absolute_permeability = 1/(total_resistance)

    # Estimating permeability means calculating permeability for each 
    # bootstrapped sample, and then performing averaging/std
    bs_r_profile = np.zeros((n_forceouts, n_bs))
    bs_d_profile = np.zeros((n_forceouts, n_bs))
    bs_p = np.zeros(n_bs)
    for bs_i in range(n_bs):
        for window in range(n_forceouts):
           bs_d_profile[window, bs_i] = RT2/bs_int_facf[window,bs_i]
           bs_r_profile[window, bs_i] = np.exp(bs_g_profile[window, bs_i]/(kb*T)) / bs_d_profile[window,bs_i]
        bs_p[bs_i] = 1/(np.sum(bs_r_profile[:, bs_i]) * dz * 1e-8)

    absolute_permeability = np.mean(bs_p)
    absolute_permeability_err = np.std(bs_p)


        
    print("{0},{1},{2},{3},{4},{5},{6},{7}".format(n_bs, bs_sample_size, 
        d_from_local_i_permeability, d_from_local_i_permeability_err,
        d_from_leaflet_i_permeability, d_from_leaflet_i_permeability_err,
        absolute_permeability, absolute_permeability_err))

    
    ##########
    ### Plotting ####
    ##########
    
    # mean forces
    fig = plt.figure(1)
    x_vals = np.arange(0, 0.2*absolute_p.shape[0], 0.2)
    plt.errorbar(x_vals, absolute_p.mean_force.values, 
            yerr=absolute_p.mean_force_std.values,
            label='absolute coordinates')
    for interface in leaflet_interfaces:
        plt.axvline(x=interface, color='r')
    plt.ylabel("Mean force (kcal/$\AA$ mol)", fontsize=20)
    plt.xlabel("Distance (nm)", fontsize=20)
    plt.legend()
    plt.savefig("absolute_xyz_meanf.jpg")
    plt.close()
    
    fig = plt.figure(1)
    plt.errorbar(d_from_local_i_p.bin_center.values, d_from_local_i_p.mean_force.values, 
            yerr = d_from_local_i_p.mean_force_std.values, 
            label="Distance from local interface")
    plt.ylabel("Mean force (kcal/$\AA$ mol)", fontsize=20)
    plt.xlabel("Distance (nm)", fontsize=20)
    np.savetxt('dfromleafleti_meanf.dat', np.column_stack((d_from_local_i_p.bin_center.values,
        d_from_local_i_p.mean_force.values, d_from_local_i_p.mean_force_std.values)))
    plt.legend()
    plt.savefig("dfromi_xyz_meanf.jpg")
    plt.close()
    
    # free energy
    fig = plt.figure(1)
    plt.errorbar(x_vals, absolute_p.g_mean.values, 
            yerr=absolute_p.g_err.values,
            label='absolute coordinates')
    for interface in leaflet_interfaces:
        plt.axvline(x=interface, color='r')
    plt.ylabel("Free energy (kcal/mol)", fontsize=20)
    plt.xlabel("Distance (nm)", fontsize=20)
    plt.legend()
    plt.savefig("absolute_xyz_fe.jpg")
    plt.close()
    
    fig = plt.figure(1)
    plt.errorbar(d_from_local_i_p.bin_center.values, d_from_local_i_p.g_mean.values, 
            yerr=d_from_local_i_p.g_err.values,
            label='Distance from local interface')
    plt.ylabel("Free energy (kcal/mol)", fontsize=20)
    plt.xlabel("Distance (nm)", fontsize=20)
    np.savetxt('dfromleafleti_fe.dat',np.column_stack((d_from_local_i_p.bin_center.values, 
        d_from_local_i_p.g_mean.values, d_from_local_i_p.g_err.values)))
    plt.legend()
    plt.savefig("dfromi_xyz_fe.jpg")
    plt.close()
    
    
    
    # Diffusion
    fig = plt.figure(1)
    plt.errorbar(x_vals, absolute_p.d_mean.values, 
            yerr=  absolute_p.d_err.values,
            label='absolute coordinates')
    for interface in leaflet_interfaces:
        plt.axvline(x=interface, color='r')
    
    plt.yscale("log")
    plt.ylabel("Diffusion (cm$^2$/sec)", fontsize=20)
    plt.xlabel("Distance (nm)", fontsize=20)
    plt.legend()
    plt.savefig('absolute_xyz_diff.jpg')
    plt.close()
    
    fig = plt.figure(1)
    plt.errorbar(d_from_local_i_p.bin_center.values, d_from_local_i_p.d_mean.values, 
            yerr = d_from_local_i_p.d_err.values,
            label='Distance from local interface')
    plt.yscale("log")
    plt.ylabel("Diffusion (cm$^2$/sec)", fontsize=20)
    plt.xlabel("Distance (nm)", fontsize=20)
    plt.legend()
    plt.savefig('dfromi_xyz_diff.jpg')
    np.savetxt('dfromleafleti_diff.dat', np.column_stack((d_from_local_i_p.bin_center.values,
        d_from_local_i_p.d_mean.values, d_from_local_i_p.d_err.values)))
    plt.close()
    
    
    

def _select_bootstrap_indices(sample_size, n_bootstraps, bootstrap_sample_size):
    """ Return an array of indices for bootstrapping
    where each row is the set of indices belonging to a single bootstrap sample
    so number of rows is the number of bootstraps
    and number of columns is the size of each bootstrap"""
    if bootstrap_sample_size is None:
        bootstrap_sample_size = sample_size
    bootstrap_indices = np.zeros((n_bootstraps, bootstrap_sample_size))
    possible_indices = np.arange(sample_size,dtype=int)
    for i in range(bootstrap_indices.shape[0]):
        bootstrap_indices[i,:] = np.random.choice(possible_indices, 
                size=bootstrap_sample_size,replace=True)
    return bootstrap_indices


    
if __name__ == "__main__":
    n_bs_array = [1,5, 10, 50, 100, 500, 1000, 5000, 10000,50000]
    #bs_sample_size_array = [1,5,10,15,20,25,30]
    #for n_bs, bs_sample_size in itertools.product(n_bs_array, bs_sample_size_array):
    #    main(n_bs=n_bs, bs_sample_size=bs_sample_size)
    for n_bs in n_bs_array:
        main(n_bs=n_bs, bs_sample_size=None)
