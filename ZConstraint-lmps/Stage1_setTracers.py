#from __future__ import division
import numpy as np
import os
import pdb
import random

y0=-34
dy=2
NWin=35
NTracer = 7

#os.system('rm *.o*')

nodes = open('y0list.txt', 'w')
yi=y0
for i in range(NWin):
    nodes.write('%.1f \n' % yi)
    yi += dy
nodes.close()

tracers = open('tracerIDs.txt','w')

Dy = dy*NWin/NTracer
yi=y0
var_str = ''
for i in range(NTracer):
    var_str += '-var zt%d %.2f ' % (i+1, yi)
    yi += Dy

Nmin = 128*11
Nmax = 128*21
IDs = random.sample(range(Nmin,Nmax),NTracer)

for i in range(NTracer):
    var_str += '-var ID%s %d ' % (str(i+1), IDs[i])
    tracers.write('%d \n' % IDs[i])
tracers.close()

print(var_str)

#os.system ("cp %s %s" % ('restart_z33', 'restartStage1'))
outputfile = 'outStage1'
restartfile = 'restartfile'
datafile = 'data.txt'
#os.system('%s %d %s -sf gpu -pk gpu 1 -in %s %s -var %s %s' %
#         ('aprun -n', 16, 'lmp_titan', 'inStage1_setTracers.dat', var_str, 'restartfile', restartfile))
#os.system('%s %d %s -in %s %s -var %s %s' %
#         ('mpirun -n', 16, 'lmp_gpu', 'inStage1_setTracers.dat', var_str, 'restartfile', restartfile))

os.system('%s %d %s -sf gpu -pk gpu 1 -in %s %s -var %s %s' %
         ('aprun -n', 16, 'lmp_titan', 'inStage1_setTracers.dat', var_str, 'datafile', datafile))
