import numpy as np
import pdb
import mdtraj
import simtk.unit as u
import bilayer_analysis_functions
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import plot_ay
plot_ay.setDefaults()

kb = u.BOLTZMANN_CONSTANT_kB 
#kb = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA
T = 305 * u.kelvin
#traj = mdtraj.load('trajectory.dcd', top='npt.gro')
areas = []
n_frames = 0
for chunk in mdtraj.iterload('trajectory.dcd', top='npt.gro'):
    area_chunk = chunk.unitcell_lengths[:,0] * chunk.unitcell_lengths[:,1]
    areas = np.concatenate((areas, area_chunk))
    n_frames += chunk.n_frames
areas *= u.nanometer**2
#areas = traj.unitcell_lengths[:,0] * traj.unitcell_lengths[:,1] * u.nanometer**2
#areas, _= bilayer_analysis_functions.block_avg(traj, areas._value)*areas.unit
avg_area = np.mean(areas)
mean_sq_fluc = np.sum((areas - avg_area)**2)/len(areas)
ka = kb*T*avg_area/mean_sq_fluc
print(avg_area/64)
print(ka.in_units_of(u.dyne/u.centimeter))
print(ka.in_units_of(u.newton/u.meter))

#for i, frame in enumerate(traj):
#    frame.time = 10*i

# we have 10,000 frames for 100ns
# we now have 50,000 frames for 500ns
# Let's declare block size
block_sizes = [100,200,300,400, 500,600,700,800,900, 1000, 2000, 5000,
            10000, 20000, 25000, n_frames]
fig, ax = plt.subplots(1,2)
for block_size in block_sizes:
    n_blocks = int(n_frames / block_size)
    all_ka = []
    for i in range(n_blocks):
        window_areas = areas[i : i + block_size]
        window_avg = np.mean(window_areas)
        window_fluc = np.sum((window_areas - window_avg)**2)/len(window_areas)
        ka = kb*T*window_avg/window_fluc
        all_ka.append(ka.in_units_of(u.dyne/u.centimeter)._value)
    mean = np.mean(all_ka)
    #se = np.std(all_ka)/np.sqrt(n_blocks)
    se = np.std(all_ka)
    ax[0].scatter([block_size], [se])
    ax[1].scatter([block_size], [mean])
    print("{} blocksize, {} mean, {} se".format(block_size, mean, se))
ax[0].set_xlabel("Block size (frames)")
ax[0].set_ylabel("Standard error")
ax[1].set_ylabel("Compressibilty Modulus (dyne/cm)")
ax[1].set_xlabel("Block size (frames)")
fig.tight_layout()
fig.savefig('block_comparison.png')
plt.close(fig)

# Look at autocorrelations 
# 10000 frames for 100ns, we have
# 50000 frames for 500ns
corr_length = 25000
dt = 10
time_origins = np.arange(0, n_frames - corr_length, step=dt)
all_corrs = []
for origin in time_origins:
    avg_area = np.mean(areas[origin:origin+corr_length])
    area_corr = (((areas[origin]-avg_area)*(areas[origin: origin+corr_length]-avg_area)) /
                    ((areas[origin]-avg_area)**2))
    all_corrs.append(area_corr)
all_corrs = np.asarray(all_corrs)
time_vals = np.arange(0, corr_length)/100
fig, ax = plt.subplots(1,1)
ax.plot(time_vals, np.mean(all_corrs, axis=0))
fig.tight_layout()
fig.savefig('correlations.png')
plt.close(fig)

fig, ax = plt.subplots(1,1)
ax.plot(np.arange(0, len(areas)), areas)
ax.set_xlabel("Frames")
ax.set_ylabel("Box Area")
ax.grid()
fig.tight_layout()
fig.savefig('areas.png')
plt.close(fig)

