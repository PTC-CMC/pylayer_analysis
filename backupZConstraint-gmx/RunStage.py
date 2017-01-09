import sys
import os
from optparse import OptionParser
'''
RunStage1_Weak:
    Navigates through each Sim directory, running the appropriate job submission script
'''

cwd = os.getcwd()

parser = OptionParser()
parser.add_option('--pbs', action = "store_true", dest = 'pbs_script')
parser.add_option('--slurm', action = "store_true", dest = 'slurm_script')
parser.add_option('--start', action = "store_true", dest = 'start_sims')
parser.add_option('--cont', action = 'store_true', dest = 'cont_sims')
parser.add_option('--stage', action = 'store', type = 'string', dest = 'stage')
(options, args) = parser.parse_args()

pbs_script = options.pbs_script
slurm_script = options.slurm_script
start_sims = options.start_sims
cont_sims = options.cont_sims
filename = ('Stage' + str(options.stage))

if len(str(options.stage)) == 0:
    print ('Specify filename')
    sys.exit()

if not ( pbs_script or slurm_script):
    print ('Specify pbs or slurm')
    sys.exit()
if not (start_sims or cont_sims):
    print ('Specify start or continue')
    sys.exit()

#Loop through current directory, looking for "Sim" folders, if this is a slurm file
if slurm_script:
    for file in os.listdir('.'):
        if os.path.isdir(file) and 'Sim' in file:
            #If we've found a "sim" folder, start looking in that folder for the correct job script
            for file2 in os.listdir('.' + '/' + file):
                #Look for pbs continue and start scripts according to filename
                if cont_sims:
                    if filename in file2 and "_cont.sbatch" in file2:
                        #os.sys('qsub {}'.format(file2))
                        print (file +'/' + file2)
                    else:
                        pass
    
                elif start_sims:
                    if filename in file2 and "_start.sbatch" in file2:
                        #os.sys('qsub {}'.format(file2))
                        print (file + '/' + file2)
                    else:
                        pass
    
                else:
                    print('Cannot find script')
    
        else:
            pass

if pbs_script:
    for file in os.listdir('.'):
        if cont_sims and filename in file and '_cont.pbs' in file:
            #os.sys('qsub {}'.format(file))
            print(file)
        elif start_sims and filename in file and '_start.pbs' in file:
            #os.sys('qsub {}'.format(file))
            print(file)
            #file1 = open(file[:-10]+'.gro', 'w')
            #file1.close()
            #file2 = open(file[:-10]+'.tpr', 'w')
            #file2.close()
        else:
            pass


#os.sys('qstat -u ahy3nz')
