import pdb
import os
import numpy as np
import sys
import mdtraj
import itertools
import bilayer_analysis_functions
import grid_analysis
import matplotlib
import matplotlib.pyplot as plt
import glob

def main():
    curr_dir = os.getcwd()
    all_sweeps = [thing for thing in os.listdir() if os.path.isdir(thing) and 'sweep' in thing]
    xbin_centers = None
    ybin_centers = None
    xedges = None
    yedges = None
    all_bot_surfaces = []
    all_top_surfaces = []
    all_histos = []
    for sweep in all_sweeps:
        print(sweep)
        os.chdir(os.path.join(curr_dir,sweep))
        all_sims = [thing for thing in os.listdir() if os.path.isdir(thing) and 'Sim' in thing]
        for sim in all_sims:
            os.chdir(os.path.join(curr_dir, sweep, sim))
            (xbin_centers, ybin_centers, xedges, yedges, bot_surface, 
                    top_surface, histo) = run_routine(xbin_centers, ybin_centers,
                                                    xedges,yedges)
            all_bot_surfaces.append(bot_surface)
            all_top_surfaces.append(top_surface)
            all_histos.append(histo)

    os.chdir(curr_dir)
    avg_bot_surface = np.mean(all_bot_surfaces, axis=0)
    avg_top_surface = np.mean(all_top_surfaces, axis=0)
    avg_histo = np.mean(all_histos, axis=0)
    avg_histo /= np.sum(avg_histo)
    generate_plots(avg_bot_surface, avg_top_surface, avg_histo, xbin_centers, ybin_centers)

def get_tracer_atoms(traj, tracers):
    tracer_residues = [r for r in traj.topology.residues if r.index in tracers]
    tracer_atoms = []
    for res in tracer_residues:
        tracer_atoms.extend([a.index for a in res.atoms])
    return tracer_atoms

def get_waters(traj, tracers, midplane):
    tracer_atoms = get_tracer_atoms(traj, tracers)

    bot_water_indices = [a for a in traj.topology.select('resname HOH or resname SOL')
                        if np.mean(traj.xyz[:,a,2]) < midplane 
                        and a not in tracer_atoms]
    top_water_indices = [a for a in traj.topology.select('resname HOH or resname SOL')
                        if np.mean(traj.xyz[:,a,2]) > midplane
                        and a not in tracer_atoms]

    return bot_water_indices, top_water_indices


def run_routine(xbin_centers, ybin_centers, xedges, yedges):
    xtcfile = glob.glob('Stage4*.xtc')[0]
    grofile = glob.glob('Stage4*.gro')[0]
    traj = mdtraj.load(xtcfile, top=grofile)
    tracers = np.loadtxt('tracers.out', dtype=int)
    midplane = traj.unitcell_lengths[-1,2]/2
    grid_size = 1
    thickness = 0.5
    headgroup_indices = grid_analysis._get_headgroup_indices(traj)
    bot_water_indices, top_water_indices = get_waters(traj, tracers, midplane)



    if xbin_centers is None:
        _, xbin_centers, ybin_centers, xedges, yedges = grid_analysis.calc_density_surface(
            traj, headgroup_indices, grid_size=grid_size, thickness=thickness)
        
    xbin_width = xbin_centers[1] - xbin_centers[0]
    ybin_width = ybin_centers[1] - ybin_centers[0]
    top_surface = np.zeros((len(xbin_centers), len(ybin_centers)))
    bot_surface = np.zeros((len(xbin_centers), len(ybin_centers)))
    top_indices = np.zeros((len(xbin_centers), len(ybin_centers)))
    bot_indices = np.zeros((len(xbin_centers), len(ybin_centers)))

    #####################
    ## Identifying 'channeling' regions grid-by-grid
    ####################
    # *_surface cooresponds to the time-averaged position closest to the midplane in that region
    # *_indices corresponds to the atomic index that is closest to the midplane in that region
    for i, x in enumerate(xbin_centers):
        for j, y in enumerate(ybin_centers):
            atoms_xy = grid_analysis._find_atoms_within(traj, x=x, y=y, 
                    atom_indices=top_water_indices, 
                    xbin_width=xbin_width, ybin_width=ybin_width)
            # First compute distances from midplane
            # This is n_frame x n_atoms
            dist_from_midplane = abs(traj.xyz[:, atoms_xy, 2] -  midplane)
            mins = []
            # For each frame, find the 10 deepest atoms
            # Store the average z coordinate of those 10 deepest atoms
            # Mins is n_frame x 1
            for row in dist_from_midplane:
                min_indices = row.argsort()[:10]
                mins.append(np.mean([row[arg] for arg in min_indices]))
            # Compute the trajectory average of the 10-averaged atoms
            # This is a single number
            avg_mins = np.mean(np.array(mins))

            top_surface[i,j] = avg_mins
            top_indices[i,j] = min_indices[0]


            atoms_xy = grid_analysis._find_atoms_within(traj, x=x, y=y, 
                    atom_indices=bot_water_indices, 
                    xbin_width=xbin_width, ybin_width=ybin_width)
            # First compute distances from midplane
            # This is n_frame x n_atoms
            dist_from_midplane = abs(traj.xyz[:, atoms_xy, 2] -  midplane)
            mins = []
            # For each frame, find the 10 deepest atoms
            # Store the average z coordinate of those 10 deepest atoms
            # Mins is n_frame x 1
            for row in dist_from_midplane:
                min_indices = row.argsort()[:10]
                mins.append(np.mean([row[arg] for arg in min_indices]))
            # Compute the trajectory average of the 10-averaged atoms
            # This is a single number
            avg_mins = np.mean(np.array(mins))

            bot_surface[i,j] = avg_mins
            bot_indices[i,j] = min_indices[0]



    #####################
    ## Counting tracer occupancy
    #####################
    tracer_atoms = get_tracer_atoms(traj, tracers)
    histo = np.zeros((len(xbin_centers), len(ybin_centers)))
    for tracer_index in tracer_atoms[::3]:
        foo = np.histogram2d(traj.xyz[:,tracer_index,0], traj.xyz[:, tracer_index,1],
                bins=[xedges, yedges])
        histo += foo[0]
    #histo = histo/np.sum(histo)
    return (xbin_centers, ybin_centers, xedges, yedges, 
            bot_surface, top_surface, histo)




def generate_plots(bot_surface, top_surface, histo, xbin_centers, ybin_centers):
    ###################
    ## Plotting
    ##################
    fig, ax = plt.subplots(2,2, figsize=(10,8))
    vmin = np.min([np.min(top_surface), np.min(top_surface)])
    vmax = np.max([np.max(top_surface), np.max(top_surface)])
    im =ax[0,0].imshow(top_surface.T, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0,0].set_xticks(range(len(xbin_centers)))
    ax[0,0].set_xticklabels(np.round(xbin_centers,2))
    ax[0,0].set_yticks(range(len(ybin_centers)))
    ax[0,0].set_yticklabels(np.round(ybin_centers,2))

    ax[0,0].set_title("Top leaflet water distance from midplane")
    fig.colorbar(im, ax=ax[0,0])

    im =ax[0,1].imshow(bot_surface.T, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0,1].set_xticks(range(len(xbin_centers)))
    ax[0,1].set_xticklabels(np.round(xbin_centers,2))
    ax[0,1].set_yticks(range(len(ybin_centers)))
    ax[0,1].set_yticklabels(np.round(ybin_centers,2))

    ax[0,1].set_title("Bot leaflet water distance from midplane")
    fig.colorbar(im, ax=ax[0,1])

    im =ax[1,0].imshow(histo.T, cmap='viridis')
    ax[1,0].set_xticks(range(len(xbin_centers)))
    ax[1,0].set_xticklabels(np.round(xbin_centers,2))
    ax[1,0].set_yticks(range(len(ybin_centers)))
    ax[1,0].set_yticklabels(np.round(ybin_centers,2))
    ax[1,0].set_title("Tracer Fractional Occupation")
    fig.colorbar(im, ax=ax[1,0])

    ax[1,1].set_visible(False)
    fig.tight_layout()
    fig.savefig('domain_passage.png')

if __name__ == "__main__":
    main()
