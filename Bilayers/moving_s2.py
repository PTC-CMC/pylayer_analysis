import numpy as np
import mdtraj
import bilayer_analysis_functions

######################
## Moving s2
## Examine S2 along segments of tail chains
######################

def main():
    traj = mdtraj.load('npt_80-100ns.xtc', top='npt.gro')
    results = moving_s2_routine(traj, forcefield='charmm36', window_size=3)

def moving_s2_routine(traj, forcefield='charmm36', window_size=3):
    """ For each residue compute a moving S2 value

    Parameters
    ---------
    traj : MTraj trajectory
    forcefield : str, default 'charmm36'
    window_size : int, default 3
    Returns
    -------
    results : dict
        keys: residue indices
            For one-tailed molecules, this is simple
            For two-tailed molecules (PC), this is done by adding 'a' to the key
        values: dict
            keys: 's2_mean' and 's2_err'
            values: respective quantities, down the chain.
                i.e. results[resid]['s2_mean'][3] gets the s2 for resid
                between carbons [3, 3 + window_size)
    """

    
    tail_groups, head_groups = bilayer_analysis_functions.identify_groups(traj,
            forcefield='charmm36')
    unique_resnames = set(r.name for r in traj.topology.residues 
            if not r.is_water and 'PC' not in r.name)
    pc_resnames = set(r.name for r in traj.topology.residues 
            if not r.is_water and 'PC' in r.name)

    results = {}
    for resname in unique_resnames:
        selected_keys = _select_tails(traj, tail_groups, resname, tail=None)
        all_s2 = calc_moving_s2(traj, tail_groups, selected_keys, 
                window_size=3)
        results[resname] = {'s2_mean': np.mean(all_s2, axis=0),
                            's2_err': np.std(all_s2, axis=0)}


    for pc in pc_resnames:
        selected_keys =_select_tails(traj, tail_groups, pc, tail='a')
        all_s2 = calc_moving_s2(traj, tail_groups, selected_keys, window_size=3)
        results[pc+'a'] = {'s2_mean': np.mean(all_s2, axis=0),
                            's2_err': np.std(all_s2, axis=0)}

        selected_keys =_select_tails(traj, tail_groups, pc, tail='b')
        all_s2 = calc_moving_s2(traj, tail_groups, selected_keys, window_size=3)
        results[pc+'b'] = {'s2_mean': np.mean(all_s2, axis=0),
                            's2_err': np.std(all_s2, axis=0)}

    return results

def calc_moving_s2(traj, tail_groups, selected_keys, window_size=3):
    """ Compute S2 over a moving window so you can look at different segments

    Parameters
    ----------
    traj : MDTraj trajectory
    tail_groups : dict
        Keys: residue IDs
        Values: atom indices 
    selected_keys: list
        residue IDs so we can pick out specific residue chains
    window_size: int, default 3
        number of consecutive carbons to look at for S2

    Returns
    -------
    all_s2: array, traj.n_frames x n_carbons - window_size
        Each row is the nematic order for a frame
        Each column is the nematic order for that selection of chains.
            i.e. column 0 looks at the S2 for carbon[0,window_size)
    """
    n_carbons = len([a for a in tail_groups[selected_keys[0]] if 'H' != traj.topology.atom(a).name[0]])
    all_s2 = np.zeros((traj.n_frames, n_carbons-3))
    for i, carbon_start in enumerate(range(n_carbons-3)):
        tail_no_h = {key: [int(a) for a in tail_groups[key] 
            if 'H' != traj.topology.atom(a).name[0]] for key in tail_groups.keys()}
        sub_tails = [tail_no_h[key][carbon_start:carbon_start+3] 
                for key in selected_keys]
        all_s2[:, i] = mdtraj.compute_nematic_order(traj, indices=sub_tails)
    return all_s2


def _select_tails(traj, tail_groups, resname, tail=None):
    """ pick out specific chains

    Returns
    -------
    selected_keys : list
        The keys in tail_groups that correspond to that resname

    tail: str, default None
        If 'a' or 'b', look at that particular tail
    """
    selected_keys = []
    for key in tail_groups.keys():
        # Snip out the 'a' or 'b' extensions to find resid
        resid = key.replace('a', '')
        resid = int(resid.replace('b', ''))
        if tail is None:
            if traj.topology.residue(resid).name == resname:
                selected_keys.append(key)
        if tail == 'a':
            if traj.topology.residue(resid).name == resname and 'a' in key:
                selected_keys.append(key)
        if tail == 'b':
            if traj.topology.residue(resid).name == resname and 'b' in key:
                selected_keys.append(key)

    return selected_keys

if __name__ == "__main__":
    main()

