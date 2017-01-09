from ParallelPullingSetupSystem import *
import numpy as np
from optparse import OptionParser
import os

#Continue from the stage 1 weak pulling simulation, but setting up for a strong pulling simulation
parser = OptionParser()
parser.add_option('--t', action = 'store', type = 'string', dest = 'tracerfile', default = 'tracers.out')
parser.add_option('--z', action = 'store', type = 'string', dest = 'zwindows', default = 'z_windows.out')
parser.add_option('--gro', action = 'store', type 'string', dest = 'grofile', default = 'something.gro')
parser.add_option('--k', action = 'store', type = 'float', dest = 'pull_coord_k', default = '40')
(options,args) = parser.parse_args()

thing = ParallelPullingSetupSystem()
tracerlist_filename = options.tracerfile
zwindows_filename = options.zwindows

#Read tracerlist
tracerlist = open(tracerlist_filename, 'r')
tracerlistlines = tracerlist.readlines()
thing.read_tracers(tracerlistlines)

#Read Zwindows
zwindows = open(zwindows_filename, 'r')
zwindowslines = zwindows.readlines()
thing.read_zlist(zwindowslines)
