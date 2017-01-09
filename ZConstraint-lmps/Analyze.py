import permeability as prm
import matplotlib.pyplot as plt
import pickle as pickle
from os import system 
import pdb

#data_dir = '/Users/rmhartkamp/Dropbox/PostDoc_2_Vanderbilt/Simulation/Permeability/DSPC_C12OH_3_1'
data_dir = '/raid6/homes/ahy3nz/Programs/Analysis/ZConstraint-lmps'

prm.analyze_sweeps(data_dir, timestep=2.0, verbosity=2, directory_prefix='Forces')
#prm.analyze_sweeps(data_dir, timestep=2.0, verbosity=2, directory_prefix='Sweep44')
#prm.analyze_sweeps(data_dir, timestep=2.0, verbosity=2, directory_prefix='Sweep35')
#prm.analyze_sweeps(data_dir, timestep=2.0, verbosity=2, directory_prefix='Sweep36')
#prm.analyze_sweeps(data_dir, timestep=2.0, verbosity=2, directory_prefix='Sweep34')

pdb.set_trace()

#forcetime = prm.force_timeseries(data_dir, timestep=2.0, n_windows=5, start_window=15, directory_prefix='Sweep')
#prm.plot_timeseries(forcetime['time'], forcetime['forces']) 
#system('open -a preview *.png')


#output = prm.analyze_force_acf_data(data_dir, 305.0, timestep=2.0, verbosity=1)
output = prm.analyze_force_acf_data(data_dir, 305.0, timestep=2.0, verbosity=1, directory_prefix='Forces')
pickle.dump(output, open('output.p', 'wb'))

#output = pickle.load(open('output.p', 'rb'))
#pdb.set_trace()

system('rm *.pdf *.png')

prm.plot_forces(output['z'], output['forces'], fig_filename='forces.pdf')
prm.plot_free_energy_z(output['z'], output['dG'], fig_filename='delta_G.pdf')
prm.plot_force_acfs_time(output['time'], output['facf_windows'], normalize=True)
prm.plot_int_acfs_time(output['time'], output['int_facf_windows'])
prm.plot_symmetrized_free_energy(output['z'], output['dG_sym'], 
        output['dG_sym_err'],savefig=True)
prm.plot_sym_diffusion_coefficient_z(output['z'], output['d_z_sym'], 
        output['d_z_sym_err'],savefig=True)
prm.plot_resistance_z(output['z'], output['R_z'], output['R_z_err'], savefig=True)
prm.plot_sym_exp_free_energy(output['z'], output['dG_sym'], output['dG_sym_err'], output['d_z_sym'], 
        output['d_z_sym_err'], output['R_z'], output['R_z_err'], 305)
#system('open -a preview *.pdf *.png')
