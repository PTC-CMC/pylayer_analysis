import numpy as np
import json
from collections import OrderedDict
""" Group defintiions for several lipid species
depends on the forcefield and molecule parameterization """

# //////////////////
# Gromos 53a6 Lipids
# //////////////////
g53a6_groups = OrderedDict()
########################
### PC phospholipids ###
########################
g53a6_groups['DPPC'] = OrderedDict()
g53a6_groups['DPPC']['head'] = [6,7,8,9,10]
g53a6_groups['DPPC']['tail_1'] = [14,16,17,18,19,20,21,22,23,24,25,26]
g53a6_groups['DPPC']['tail_2'] = [33,35,36,37,38,39,40,41,42,43,44,45] 

g53a6_groups['DSPC'] = OrderedDict()
g53a6_groups['DSPC']['head'] = [6,7,8,9,10]
g53a6_groups['DSPC']['tail_1'] = [14,16,17,18,19,20,21,22,23,24,25,26]
g53a6_groups['DSPC']['tail_2'] = [35,37,38,39,40,41,42,43,44,45,46,47]

##################
### Emollients ###
##################
g53a6_groups['ISIS'] = OrderedDict()
g53a6_groups['ISIS']['head'] = [18,19,20]
g53a6_groups['ISIS']['tail_1'] = [0,1,2,3,4,5,6,6,7,8,9,10,11,12,13,14,15,16,17]
g53a6_groups['ISIS']['tail_2'] = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]


##################
### FFA groups ###
##################
g53a6_groups['acd12'] = OrderedDict()
g53a6_groups['acd12']['tail'] = [0,1,2,3,4,5,6,7,8,9,10]
g53a6_groups['acd12']['head'] = [11,12,13,14]

g53a6_groups['acd14'] = OrderedDict()
g53a6_groups['acd14']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12]
g53a6_groups['acd14']['head'] = [13,14,15,16]

g53a6_groups['acd16'] = OrderedDict()
g53a6_groups['acd16']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
g53a6_groups['acd16']['head'] = [15,16,17,18]

g53a6_groups['acd18'] = OrderedDict()
g53a6_groups['acd18']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
g53a6_groups['acd18']['head'] = [17,18,19,20]

g53a6_groups['acd20'] = OrderedDict()
g53a6_groups['acd20']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
g53a6_groups['acd20']['head'] = [19,20,21,22]

g53a6_groups['acd22'] = OrderedDict()
g53a6_groups['acd22']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
g53a6_groups['acd22']['head'] = [21,22,23,24]

g53a6_groups['acd24'] = OrderedDict()
g53a6_groups['acd24']['tail'] =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
g53a6_groups['acd24']['head'] = [23,24,25,26]

#################
### OH groups ###
#################
g53a6_groups['alc12'] = OrderedDict()
g53a6_groups['alc12']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11]
g53a6_groups['alc12']['head'] = [12,13]

g53a6_groups['alc14'] = OrderedDict()
g53a6_groups['alc14']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
g53a6_groups['alc14']['head'] = [14,15]

g53a6_groups['alc16'] = OrderedDict()
g53a6_groups['alc16']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
g53a6_groups['alc16']['head'] = [16,17]

g53a6_groups['alc18'] = OrderedDict()
g53a6_groups['alc18']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
g53a6_groups['alc18']['head'] = [18,19]

g53a6_groups['alc20'] = OrderedDict()
g53a6_groups['alc20']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
g53a6_groups['alc20']['head'] = [20,21]

g53a6_groups['alc22'] = OrderedDict()
g53a6_groups['alc22']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
g53a6_groups['alc22']['head'] = [22,23]

g53a6_groups['alc24'] = OrderedDict()
g53a6_groups['alc24']['tail'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
g53a6_groups['alc24']['head'] = [24,25]


# ///////////////
# Charmm36 Lipids
# ///////////////
c36_groups = OrderedDict()

# #######################
# ### PC Phosphoipids ###
# #######################
c36_groups['DPPC'] = OrderedDict()
c36_groups['DPPC']['head'] = [19,20,21,22,23]
c36_groups['DPPC']['tail_1'] = [39,41,87,90,93,96,99,102,105,108,111,114,117,120,123,
        126]
c36_groups['DPPC']['tail_2'] = [30,32,44,47,50,53,56,59,62,65,68,71,74,77,80,83]


c36_groups['DSPC'] = OrderedDict()
c36_groups['DSPC']['head'] = [19,20,21,22,23]
c36_groups['DSPC']['tail_1'] = [39,41,93,96,99,102,105,108,111,114,117,120,123,126,
        129,132,135,138]
c36_groups['DSPC']['tail_2'] = [30,32,44,47,50,53,56,59,62,65,68,71,74,77,80,83,86,
        89]

c36_groups['DOPC'] = OrderedDict()
c36_groups['DOPC']['head'] = [19,20,21,22,23]
c36_groups['DOPC']['tail_1'] = [39,41,91,94,97,100,103,106,109,111,113,116,119,
        122,125,128,131,134]
c36_groups['DOPC']['tail_2'] = [30,32,44,47,50,53,56,59,62,64,66,69,72,75,78,81,84,
        87]

c36_groups['ISIS'] = OrderedDict()
c36_groups['ISIS']['head'] = [0,1,2]
c36_groups['ISIS']['tail_1'] = np.arange(109, 54, step=-1)
c36_groups['ISIS']['tail_2'] = np.concatenate(([1], 
    np.arange(3,55,step=1)))


# #################
# ### Ceramides ###
# #################
# Sphingosine tails are tail_1
# FA tails are tail_2

c36_groups['cer1'] = OrderedDict()
c36_groups['cer1']['head'] = [24, 25, 22, 23, 18, 105, 19, 106, 107, 20, 21, 15, 
                              104, 16, 17, 14, 103, 13, 102]
c36_groups['cer1']['tail_1'] = np.arange(0,13)
c36_groups['cer1']['tail_2'] = np.concatenate(( np.arange(26,55), np.arange(58, 76) ))

c36_groups['ecer2'] = OrderedDict()
c36_groups['ecer2']['head'] = [ 0, 1, 2, 3, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
                                59, 60, 61, 62, 63, 64]
c36_groups['ecer2']['tail_1'] = np.arange(65,105)
c36_groups['ecer2']['tail_2'] = np.arange(11,50)

c36_groups['ecer3'] = OrderedDict()
c36_groups['ecer3']['head'] = [ 0, 1, 2, 3, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
                                59, 60, 61, 62, 63, 64]
c36_groups['ecer3']['tail_1'] = np.arange(65,105)
c36_groups['ecer3']['tail_2'] = np.arange(11,50)


c36_groups['ucer2'] = OrderedDict()
c36_groups['ucer2']['head'] = [ 0, 1, 2, 3, 74, 75, 76, 77, 78, 79, 80, 81, 82, 
                                83, 84, 85, 86, 87, 88]
c36_groups['ucer2']['tail_1'] = np.arange(89, 129)
c36_groups['ucer2']['tail_2'] = np.arange(11, 74)

c36_groups['ucer6'] = OrderedDict()
c36_groups['ucer6']['head'] = []
c36_groups['ucer6']['tail_1'] = []
c36_groups['ucer6']['tail_2'] = np.arange(4,75)

####################
# ### FFA Groups ###
####################

c36_groups['ffa12'] = OrderedDict()
c36_groups['ffa12']['head'] = np.arange(0,4)
c36_groups['ffa12']['tail'] = np.arange(4,38)

c36_groups['ffa16'] = OrderedDict()
c36_groups['ffa16']['head'] = np.arange(15,19)
c36_groups['ffa16']['tail'] = np.concatenate(( np.arange(0,16), np.arange(19,50) ))

c36_groups['ffa18'] = OrderedDict()
c36_groups['ffa18']['head'] = np.arange(0,4)
c36_groups['ffa18']['tail'] = np.arange(4,53, step=3)

c36_groups['ffa20'] = OrderedDict()
c36_groups['ffa20']['head'] = np.arange(0,4)
c36_groups['ffa20']['tail'] = np.arange(4,59, step=3)

c36_groups['ffa22'] = OrderedDict()
c36_groups['ffa22']['head'] = np.arange(0,4)
c36_groups['ffa22']['tail'] = np.arange(4,65, step=3)

c36_groups['ffa24'] = OrderedDict()
c36_groups['ffa24']['head'] = np.arange(23,27)
c36_groups['ffa24']['tail'] = np.concatenate(( np.arange(0,24), np.arange(27,74) )) 

###################
# ### OH Groups ###
###################

c36_groups['oh12'] = OrderedDict()
c36_groups['oh12']['head'] = [35,38]
c36_groups['oh12']['tail'] = np.concatenate(( np.arange(34,21, step=-1),
    np.arange(0,21)))

c36_groups['oh16'] = OrderedDict()
c36_groups['oh16']['head'] = [47,50]
c36_groups['oh16']['tail'] = np.concatenate(( np.arange(46, 27, step=-1), 
    np.arange(0, 28)))

c36_groups['oh18'] = OrderedDict()
c36_groups['oh18']['head'] = [55,56]
c36_groups['oh18']['tail'] = np.concatenate(( np.arange(25, -1, step=-1),
    np.arange(31, 56, step=1)))

c36_groups['oh20'] = OrderedDict()
c36_groups['oh20']['head'] = [30, 31]
c36_groups['oh20']['tail'] = np.concatenate(( np.arange(57, 31, step=-1),
    np.arange(0, 30, step=1)))

c36_groups['oh22'] = OrderedDict()
c36_groups['oh22']['head'] = [31,34]
c36_groups['oh22']['tail'] = np.concatenate(( np.arange(68, 34, step=-1),
    np.arange(0,31, step=1)))

c36_groups['oh24'] = OrderedDict()
c36_groups['oh24']['head'] = [71,74]
c36_groups['oh24']['tail'] = np.concatenate(( np.arange(70, 39, step=-1),
    np.arange(0, 40)))


###################
### Cholesterol ###
##################
c36_groups['chol'] = OrderedDict()
c36_groups['chol']['head'] = [0,1,2,3]
c36_groups['chol']['tail'] = np.arange(49,74)

c36_groups['CHL1'] = OrderedDict()
c36_groups['CHL1']['head'] = [0,1,2,3]
c36_groups['CHL1']['tail'] = np.arange(49,74)

# ///////////////
# MSIBI Lipids
# ///////////////
msibi_groups = OrderedDict()

# #######################
# ### PC Phosphoipids ###
# #######################
msibi_groups['DSPC'] = OrderedDict()
msibi_groups['DSPC']['head'] = [1]
msibi_groups['DSPC']['tail_1'] = [3, 10, 11, 12, 13, 14, 15]
msibi_groups['DSPC']['tail_2'] = [2, 4, 5, 6, 7, 8, 9]

####################
# ### FFA Groups ###
####################
msibi_groups['ffa16'] = OrderedDict()
msibi_groups['ffa16']['head'] = [0]
msibi_groups['ffa16']['tail'] = [1,2,3,4,5]


def gromos53a6_groups():
    return g53a6_groups

def charmm36_groups():
    return c36_groups

def cg_groups():
    return msibi_groups



