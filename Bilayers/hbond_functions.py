import sys
from collections import OrderedDict
import MDAnalysis as mda
""" Hydrogen bond donor and acceptor defintiions for several lipid species
depends on the forcefield and molecule parameterization """


# //////////////////
# Charmm36 Lipids
# //////////////////
c36_donors = OrderedDict()
c36_acceptors = OrderedDict()

########################
### PC phospholipids ###
########################
c36_acceptors['DSPC'] = ['O11', 'O12', 'O13', 'O14',  'O31', 'O32', 'O21', 'O22',
                        'OSLP', 'O2L', 'OSL', 'OBL']
c36_donors['DSPC'] = []

#############
### Water ###
#############
c36_acceptors['HOH'] = ['OT' , 'OW']
c36_donors['HOH'] = ['OT' , 'OW']

c36_acceptors['SOL'] = ['OT' , 'OW']
c36_donors['SOL'] = ['OT' , 'OW']



##################
### FFA groups ###
##################
c36_acceptors['ffa12'] = ['O1', 'O2', 'OHL', 'OBL']
c36_donors['ffa12'] = ['O1', 'OHL']

c36_acceptors['ffa16'] = ['O17', 'O19', 'OCL', 'OBL']
c36_donors['ffa16'] = ['O17', 'OCL']

c36_acceptors['ffa18'] = ['O1', 'O2', 'OHL', 'OBL']
c36_donors['ffa18'] = ['O1', 'OHL']

c36_acceptors['ffa20'] = ['O1', 'O2', 'OHL', 'OBL']
c36_donors['ffa20'] = ['O1', 'OHL']

c36_acceptors['ffa22'] = ['O1', 'O2', 'OHL', 'OBL']
c36_donors['ffa22'] = ['O1', 'OHL']

c36_acceptors['ffa24'] = ['O25', 'O27', 'OCL', 'OBL']
c36_donors['ffa24'] = ['O25', 'OCL']

##################
### OH groups ###
##################
c36_acceptors['oh12'] = ['O', 'OG311']
c36_donors['oh12'] = ['O', 'OG311']

c36_acceptors['oh16'] = ['O', 'OG311']
c36_donors['oh16'] = ['O', 'OG311']

c36_acceptors['oh24'] = ['O', 'OG311']
c36_donors['oh24'] = ['O', 'OG311']

def get_hbond_groups(uni, forcefield='Charmm36'):
    """ Return a list of donors and a list of acceptors atom names
    Parameters
    --------
    uni : MDAnalysis.Universe

    Returns
    -------
    c36_acceptors: list
    c36_donors: list
    """
    if 'charmm36' in forcefield.lower():
        reference_acceptors = c36_acceptors
        reference_donors = c36_donors
    else:
        sys.exit("forcefield not supported in hbond groups")
    hbond_groups = {}
    res_names = list(set(res.resname for res in uni.residues ))
    acceptor_names = []
    donor_names = []
    water_acceptor_names = []
    water_donor_names = []
    for res_name in res_names:
        if 'HOH' in res_name or 'SOL' in res_name:
            water_acceptor_names.extend(reference_acceptors[res_name])
            water_donor_names.extend(reference_acceptors[res_name])
        else:
            acceptor_names.extend(reference_acceptors[res_name])
            donor_names.extend(reference_donors[res_name])
    
    acceptor_indices = []
    donor_indices = []
    water_acceptor_indices = []
    water_donor_indices = []
    for a in uni.atoms:
        if a.name in donor_names:
            donor_indices.append(a.index)
        elif a.name in acceptor_names:
            acceptor_indices.append(a.index)
        elif a.name in water_donor_names:
            water_donor_indices.append(a.index)
        elif a.name in water_acceptor_names:
            water_acceptor_indices.append(a.index)

    hbond_groups = {'acceptor_names': acceptor_names,
                'donor_names': donor_names,
                'water_acceptor_names': water_acceptor_names,
                'water_donor_names': water_donor_names,
                'donor_indices': donor_indices,
                'acceptor_indices': acceptor_indices,
                'water_donor_indices': water_donor_indices,
                'water_acceptor_indices': water_acceptor_indices}

    return hbond_groups

def get_hbond_donor_pairs(uni, forcefield='charmm36'):
    """ Return a list of hydrogen-donor pairs

    Parameters
    ---------
    uni: MDAnalysis.Universe

    Returns
    -------
    hydrogens: list of indices
    donors: list of indices
    """
    if 'charmm36' in forcefield.lower():
        reference_acceptors = c36_acceptors
        reference_donors = c36_donors
    else:
        sys.exit("forcefield not supported in hbond groups")


    water_hydrogens = []
    water_donors = []
    hydrogens = []
    donors = []
    for bond_pair in uni.bonds:
        resname = bond_pair.atoms[0].resname
        if 'H' in bond_pair.atoms[0].name and ('O' in bond_pair.atoms[1].name or
                                            'N' in bond_pair.atoms[1].name or
                                            'F' in bond_pair.atoms[1].name):
            if 'HOH' in resname or 'SOL' in resname:
                water_hydrogens.append(bond_pair.atoms[0].index)
                water_donors.append(bond_pair.atoms[1].index)
            else:
                hydrogens.append(bond_pair.atoms[0].index)
                donors.append(bond_pair.atoms[1].index)

        elif 'H' in bond_pair.atoms[1].name and ('O' in bond_pair.atoms[0].name or
                                            'N' in bond_pair.atoms[0].name or
                                            'F' in bond_pair.atoms[0].name):
            if 'HOH' in resname or 'SOL' in resname:
                water_hydrogens.append(bond_pair.atoms[0].index)
                water_donors.append(bond_pair.atoms[1].index)
            else:
                hydrogens.append(bond_pair.atoms[0].index)
                donors.append(bond_pair.atoms[1].index)
    donor_pairs = {'water_hydrogens': water_hydrogens,
            'water_donors': water_donors,
            'hydrogens': hydrogens,
            'donors':donors}

    return donor_pairs

def get_hbond_sites():
    acceptors_dict = {'DSPC': 8,
        'ffa12':2, 'ffa16':2, 'ffa18':2, 'ffa20':2, 'ffa22':2, 'ffa24':2,
        'oh12':1, 'oh16':1, 'oh18':1, 'oh20':1, 'oh22':1, 'oh24':1,
        'HOH':2}
    donors_dict = {'DSPC': 0,
            'ffa12':1, 'ffa16':1, 'ffa18':1, 'ffa20':1, 'ffa22':1, 'ffa24':1,
            'oh12':1, 'oh16':1, 'oh18':1, 'oh20':1, 'oh22':1, 'oh24':1,
            'HOH':2}

    return donors_dict, acceptors_dict

