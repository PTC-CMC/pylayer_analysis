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

inputfile = open(options.filename,'r')
filenames = inputfile.readlines()
for i, val in enumerate(filenames):
    if "#" not in val and val.rstrip():
        val = val.strip()
        print('*********************************')
        print('{:^20s}'.format(val))
        print('*********************************')
        if options.ST:
            os.chdir('/Users/ahy3nz/Trajectories/{}'.format(val))
            os.system("echo '0' | gmx trjconv -f ST_{}.xtc -s ST_{}.tpr -pbc mol -dt 20 -o st".format(val,val))
            os.system('cp ~/Programs/Analysis/Bilayers/AnalyzeBilayer.py .')
            os.system('python AnalyzeBilayer.py -f st.xtc -c ST_{}.gro -o st'.format(val))
        else:
            os.chdir('/Users/ahy3nz/Trajectories/{}'.format(val))
            # Generate index file
            os.system("echo 'q' | gmx make_ndx -f {}.gro".format(val))
            # Add groups for the last atoms in a DSPC
            os.system("echo ' [a_53]' >> index.ndx")
            os.system("echo ' 53 ' >> index.ndx")
            os.system("echo ' [a_53]' >> index.ndx")
            os.system("echo ' 53 ' >> index.ndx")
            # trjconv and remove jumping in the raw trajectory based on atom 53
            os.system("echo '8 0' | gmx trjconv -f md_{}.xtc -s {}.gro -pbc nojump -o nojump.xtc -center -n index.ndx".format(val, val))
            # trjconv and pbc mol to move everything into box
            os.system("echo '0' | gmx trjconv -f nojump.xtc -s md_{}.tpr -pbc mol -dt 20 -o nopbc.xtc".format(val))
            
            #os.system("echo '1 0' | gmx trjconv -f md_{}.xtc -s md_{}.tpr -pbc nojump -center -dt 20 -o nopbc".format(val,val))
            # truncate last 20 ns 
            os.system("echo '0' | gmx trjconv -f nopbc.xtc -s md_{}.tpr -dt 20 -b 80000 -e 100000 \
                    -o last20".format(val,val))
            os.system('cp ~/Programs/Analysis/Bilayers/AnalyzeBilayer.py .')
            os.system('python AnalyzeBilayer.py -f nopbc.xtc -c md_{}.gro -o full'.format(val))
            os.system('python AnalyzeBilayer.py -f last20.xtc -c md_{}.gro -o last20'.format(val))
            os.system('cp ~/Programs/Utilities/Plotting/PlotXVG.py .')
            #os.system("echo '12' | gmx energy -f ST_{}.edr -o temp".format(val))
            #os.system('python PlotXVG.py -f temp.xvg')
    else:
        pass

