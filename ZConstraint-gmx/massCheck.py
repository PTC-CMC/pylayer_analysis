import numpy as np
import os
import sys
from optparse import OptionParser

"""
Given a list of filenames, check which simulations have finished
based on the respective grofile printed
"""
parser = OptionParser()
parser.add_option("--Stage", action = "store", type = "string", dest = "Stage")
parser.add_option("-f", action = "store", type = "string", dest = "filename")
parser.add_option("--ST", action="store_true", dest = "STrun")
parser.add_option("--MD", action="store_true", dest = "MDrun")
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



success_list = []
fail_list = []

for i in range(0,5):
    current_dir = os.getcwd()
    modified_dir = (current_dir + "/Sim" + str(i) + "/Stage" + extension + str(i) + ".gro")
    if os.path.isfile(modified_dir):
        success_list.append("Sim" + str(i))
    else:
        fail_list.append("Sim" + str(i))

successfile = open(str("Stage" + options.Stage + '_done.dat'),'w')
for i, val in enumerate(success_list):
    successfile.write(val+"\n")
successfile.close()

failfile = open(str("Stage" + options.Stage + '_ongoing.dat'),'w')
for i, val in enumerate(fail_list):
    failfile.write(val+"\n")
failfile.close()
