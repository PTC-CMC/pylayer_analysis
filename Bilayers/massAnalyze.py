import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser
"""
    Given a inputfile of all filenames
    Remove jumps, PBCs, analyze it, and print out the temperature over time
    Assumes certain file organization and naming 
    """

parser = OptionParser()
parser.add_option("-f", action = "store", type = "string", dest = "filename")
parser.add_option("--ST", action = "store_true", dest = "ST")
(options, args) = parser.parse_args()

#inputfile = open(options.filename,'r')
#filenames = inputfile.readlines()
filenames = ['90_WVTR_4-29c']
for i, val in enumerate(filenames):
    if "#" not in val and val.rstrip():
        val = val.strip()
        print('*********************************')
        print('{:^20s}'.format(val))
        print('*********************************')
        if options.ST:
            os.chdir('/Users/ahy3nz/Trajectories/{}'.format(val))
            # Generate index file
            os.system("echo 'q' | gmx make_ndx -f {}.gro".format(val))
            # Add groups for the last atoms in a DSPC
            os.system("echo ' [a_53]' > newindex.ndx")
            os.system("echo ' 53 ' >> newindex.ndx")
            os.system("cat index.ndx >> newindex.ndx")
            # trjconv and remove jumping in the raw trajectory based on atom 53
            os.system("echo '0 1' | gmx trjconv -f ST_{}.xtc -s {}.gro -pbc nojump -o nojump.xtc -center -n newindex.ndx".format(val, val))
            # trjconv and pbc mol to move everything into box
            os.system("echo '0' | gmx trjconv -f nojump.xtc -s ST_{}.tpr -pbc mol -dt 20 -b 10000 -e 50000 -o st.xtc".format(val))
            os.system('cp ~/Programs/Analysis/Bilayers/noblockAnalyzeBilayer.py .')
            os.system('python noblockAnalyzeBilayer.py -f ST_{}.xtc -c ST_{}.gro -o st'.format(val,val))
            os.system("rm nojump.xtc")
        else:
            os.chdir('/Users/ahy3nz/Trajectories/{}'.format(val))
            ## Generate index file
            os.system("echo 'q' | gmx make_ndx -f md_{}.gro".format(val))
            #os.system("echo 'q' | gmx make_ndx -f {}.gro".format(val))
            # Add groups for the last atoms in a DSPC
            os.system("echo ' [a_53]' > newindex.ndx")
            os.system("echo ' 53 ' >> newindex.ndx")
            os.system("cat index.ndx >> newindex.ndx")
            # trjconv and remove jumping in the raw trajectory based on atom 53
            os.system("echo '0 1' | gmx trjconv -f md_{}.xtc -s md_{}.gro -pbc nojump -o nojump.xtc -center -n newindex.ndx".format(val, val))
            # trjconv and pbc mol to move everything into box
            os.system("echo '0' | gmx trjconv -f nojump.xtc -s md_{}.tpr -pbc mol -dt 20 -o nopbc.xtc".format(val))
            os.system("echo '0' | gmx trjconv -f nojump.xtc -s md_{}.tpr -pbc mol -b 480000 -e 500000 -dt 20 -o last20.xtc".format(val))
            os.system("echo '0' | gmx trjconv -f md_{0}.gro -s md_{0}.tpr -pbc mol -o md_{0}.pdb -conect".format(val))

            
            # truncate last 20 ns 
            os.system('cp ~/Programs/Analysis/Bilayers/AnalyzeBilayer.py .')
            os.system('python AnalyzeBilayer.py -f nopbc.xtc -c md_{}.gro -o full'.format(val))
            os.system('python AnalyzeBilayer.py -f last20.xtc -c md_{}.gro -o last20'.format(val))
            os.system('rm nojump.xtc')
            os.system('rm nopbc.xtc')
            #os.system('rm last20.xtc')
    else:
        pass

