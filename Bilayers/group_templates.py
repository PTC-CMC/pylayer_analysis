import numpy as np
import json
from collections import OrderedDict
""" Group defintiions for several lipid species
depends on the forcefield and molecule parameterization """


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

 

def gromos53a6_groups():
    return g53a6_groups





