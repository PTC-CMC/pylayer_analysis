import numpy as np
import os
import sys
from optparse import OptionParser
import subprocess

parser = OptionParser()
parser.add_option("--Stage", action = "store", type = "string", dest = "Stage")
parser.add_option('--t', action = 'store', type = 'string', dest = 'tracerfile', default = 'tracers.out')
parser.add_option('--z', action = 'store', type = 'string', dest = 'zwindows', default  = 'z_windows.out')
parser.add_option('--sweep', action='store', type='int', dest='sweep')


(options, args) = parser.parse_args()

if options.Stage is "1":
    extension = "1_Weak"
elif options.Stage is "2":
    extension = "2_Strong"
elif options.Stage is "3":
    extension = "3_Moving"
elif options.Stage is "4":
    extension = "4_Eq"
elif options.Stage is "5":
    extension = "5_ZCon"
else:
    sys.exit("Invalid Stage number")

tracerlist_filename = options.tracerfile
zwindows_filename = options.zwindows
#pull_coord_rate = options.pull_coord_rate
indexfile = 'FullIndex.ndx'

#Read tracers
tracerlist = open(tracerlist_filename, 'r')
tracerlistlines = tracerlist.readlines()
N_tracer = len(tracerlistlines)


#Read zwindows
zwindows = open(zwindows_filename, 'r')
zwindowslines = zwindows.readlines()
N_window = len(zwindowslines)

N_sims = int(N_window / N_tracer)

current_dir = os.getcwd()

for i in range(N_sims):
    os.chdir("{}/sweep{}/Sim{}".format(current_dir, str(options.sweep), i))
    p = subprocess.Popen("bash Grompp_Stage{0}{1}.sh ".format(extension,i), 
            shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    print("Grompping Stage{0}{1}".format(extension,i))
    p.wait()
    #os.system("bash Grompp_Stage{}{}.sh".format(extension,i))
    if os.path.isfile((current_dir + "/sweep" + str(options.sweep) + "/Sim" + str(i) + "/Stage" + extension + str(i) + ".tpr")):
        pass
    else:
        print('Sim{} failed'.format(i))

