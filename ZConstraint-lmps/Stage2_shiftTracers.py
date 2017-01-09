from __future__ import division
import numpy as np
import os
import pdb
import random

y0 = np.loadtxt('y0list.txt')
tracerIDs = np.loadtxt('tracerIDs.txt')
dy = y0[2]-y0[1]
NWin = np.size(y0)
NTracer = np.size(tracerIDs)

var_str = ''
for i in range(NTracer):
    var_str += '-var ID%s %d ' % (str(i+1), tracerIDs[i])

print(var_str)

restartinfile = 'restart_file'
os.system ("cp %s %s" % ('restartStage1', restartinfile))
os.system('%s %d %s -pk gpu 1 -sf gpu -in %s %s -var restartfile %s' %
         ('aprun -n', 16, 'lmp_titan', 'inStage2_moveTracers.dat', var_str, restartinfile))


