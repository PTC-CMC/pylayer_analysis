from sub_zConstr import submit_Equi, submit_MDrun, printHeader
import numpy as np
import os

y0 = np.loadtxt('y0list.txt')
tracerIDs = np.loadtxt('tracerIDs.txt')
dy = y0[2]-y0[1]
NWin = np.size(y0)
NTracer = np.size(tracerIDs)
Nsims = 5
#Nsims =4

os.system("mkdir Forces")
os.system("cp y0list.txt ./Forces/.")
os.system("cp tracerIDs.txt ./Forces/.")

f1 = open('zC_Eq_sims.sh', 'w')
f2 = open('zC_MD_sims.sh', 'w')
f3 = open('zC_movefiles.sh', 'w')
printHeader(f1, nodes=Nsims, hrs=1, mins=30, job_name='sims_Eq')
printHeader(f2, nodes=Nsims, job_name='sims_MD')
printHeader(f3, nodes=1, lammpsgpu=False, hrs=0, mins=1, job_name='movingfiles')

for i in range(Nsims):
    fol = 'Sim%d' % (i+1)
    os.system("rm -rf %s" % fol)
    os.system("mkdir %s" % fol)
    os.system("cp data_sim%d.txt %s/data.txt" % (i,fol))           
    os.system("cp inStage3_equi.dat %s" % fol)
    os.system("cp inStage4_runs.dat %s" % fol)
    os.chdir(fol)
    
    var_str = ''
    winlist = np.array(range(0,NWin,int(NWin/NTracer)))
    winlist += i
    for j in range(NTracer):
        var_str += '-var ID%s %d ' % (str(j+1), tracerIDs[j])
        var_str += '-var force%d forceout%d ' % (j+1, winlist[j])
        var_str += '-var zt%d %.2f ' % (j+1, y0[winlist[j]]) 
    print(var_str)

    restartfile = 'restartfile'
    outputfile = 'outputfile'

    lammpscommand = '%s %d %s %d %s -sf gpu -pk gpu 1 -in %s %s -var %s %s &' % ('aprun -n', 16, '-N', 16, 'lmp_titan', 'inStage3_equi.dat', var_str, 'restartfile', restartfile)
    lammpscommand2 = '%s %d %s %d %s -pk gpu 1 -sf gpu -in %s %s -var %s %s &' % ('aprun -n', 16, '-N', 16, 'lmp_titan', 'inStage4_runs.dat', var_str, 'restartfile', restartfile)
    print('cd $PBS_O_WORKDIR', file=f1)    
    
    print('cd Sim%s' % str(i+1), file=f1)
    print('%s' % lammpscommand, file=f1)
    print('cd ..', file=f1)
    
    print('cd Sim%s' % str(i+1), file=f2)
    print('%s' % lammpscommand2, file=f2)
    print('cd ..', file=f2)
    
    after_commands = []
    after_commands.append("mv forceout* ../Forces/")
    after_commands.append("mv restartfile ../Forces/restartfile%s" % str(i+1))    
    after_commands.append("mv outputfile ../Forces/outputfile%s" % str(i+1))    
    after_commands.append("mv trajectory.lammps ../Forces/trajectory_%s.lammps" % str(i+1))       
    print('cd Sim%s' % str(i+1), file=f3)
    for command in after_commands:
        print(command, file=f3)
    print('cd ..', file=f3)
    
    #submit_Equi(job_name='zC_Eq_sim%s' % str(i+1), lammpscommand=lammpscommand)    
    #submit_MDrun(job_name='zC_MD_sim%s' % str(i+1), lammpscommand=lammpscommand2, after_commands=after_commands)
    
    os.chdir (os.pardir)

print('wait', file=f1)
print('wait', file=f2)

f1.close()
f2.close()
f3.close()
os.system('./RunString.sh')

