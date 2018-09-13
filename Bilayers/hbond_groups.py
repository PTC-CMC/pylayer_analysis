from collections import OrderedDict
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


##################
### FFA groups ###
##################
c36_acceptors['ffa12'] = ['O1', 'O2', 'OHL', 'OBL']
c36_donors['ffa12'] = ['O1', 'OHL']

c36_acceptors['ffa16'] = ['O17', 'O19', 'OCL', 'OBL']
c36_donors['ffa16'] = ['O17', 'OCL']

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
