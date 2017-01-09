import numpy as np
import os
import pdb
import random
import sys

'''
Pulling setup system class 
Contains methods to write out mdp files for stage 2 and 3 pulling simulations
Assumes 128 bilayer molecules and 2560 water molecules
Stage 1 to set up pulling simulations with moving references to move reference to tracer window from bulk water
Stage 2 to set up pulling simulations with fixed references to pull tracer to fixed reference at tracer window 
'''
class SystemSetup():
    def __init__(self, z0 = 1.0, dz = 0.2, N_window = 40, N_tracer = 8):
        self._z0 = z0
        self._dz = dz
        self._N_window = N_window
        self._N_tracer = N_tracer

        #Molecules [129,2688] correspond to the 2560 water molecules, but we want the water molecules in the bottom layer
        #self._tracer_list = list()
        #tracer_min = 128 * 11
        #tracer_max = 128 * 21
        #self._tracer_list = random.sample(range(tracer_min, tracer_max), self.N_tracer)
        #self.write_tracerlist(self._tracer_list)

        #Define the z-window list, starting at z0 and going up dz each time
        self._zlist = list()
        for i in range(self._N_window):
               self._zlist.append(self._z0 + (i * self._dz)) 
        self.write_zlist(self._zlist)

    @property
    def tracer_list(self):
        return self._tracer_list

    def get_Tracers(self):
        return self.tracer_list

    @property
    def z_list(self):
        return self._zlist

    def get_zlist(self):
        return self.zlist

    @property
    def z0(self):
        return self._z0

    def get_z0(self):
        return self.z0

    @property
    def dz(self):
        return self._dz

    def get_dz(self):
        return self.dz

    @z0.setter
    def z0(self, z0_new):
        self._z0 = z0_new

    def set_z0(self, z0_new):
        self.z0 = z0_new
        self.calculate_zlist()

    @dz.setter
    def dz(self, dz_new):
        self._dz = dz_new

    def set_dz(self, dz_new):
        self.dz = dz_new
        self.calculate_zlist()

    def set_N_window(self, N_window_new):
        self.N_window = N_window_new
        self.calculate_zlist()

    def calculate_zlist(self):
        self.zlist = list()
        for i in range(self.N_window):
            self.zlist.append(numpy.self.z0 + (i * self.dz))

    def set_N_tracer(self, N_tracer_new):
        self.N_tracer = N_tracer_new
        self.gather_tracer()
    
    def gather_tracer(self, grofile = None):
        #Need to make sure the tracers are on the same side of the bilayer
        #z probably less than 4
        #Tracer list is just random integers
        #tracer_min = 128 * 1
        #tracer_max = 128 * 21
        #self._tracer_list = random.sample(range(tracer_min, tracer_max), self._N_tracer)
        #self.write_tracerlist(self._tracer_list)
        tracer_min = 128 * 1
        tracer_max = 128 * 21
        tracer_list = list()
        for i in range(self._N_tracer):
            z = 8
            # Make sure we have a tracer whose z coordinate is on the bottom leaflet
            while (z > 3):
                tracerid = random.sample(range(tracer_min, tracer_max), 1)
                (x,y,z) = self.get_tracer_coordinates(grofile,tracerid[0])

            tracer_list.append(tracerid[0])
        self._tracer_list = tracer_list
        self.write_tracerlist(tracer_list)

    def read_tracers(self, tracer_list):
        self._tracer_list = list()
        for i, tracer in enumerate(tracer_list):
            self._tracer_list.append(tracer.split()[0])

    def read_zlist(self, z_list):
        #self.zlist = list()
        for i, zwindow in enumerate(z_list):
            self._zlist.append(zwindow.split()[0])
        self._dz = np.absolute(float(self._zlist[0]) - float(self._zlist[1]))
        self._z0 = self._zlist[0]

    def write_pulling_mdp(self, pull_filename, tracerlist, z_window_list, grofile, pull_coord_rate = 0,  
             pull_coord_k = 1000, posres = None, t_pulling = None, stagefive = False): 
        #Set up a pulling force on each tracer at very far windows
        #specify the group
        #Pulling reaction coordinate is just one direction (z)
        #Specify the pulling groups, 0 is a reference and 1 is the tracer
        #Set the reference position, then we will need to eliminate center of mass motion
        #pull_coord1_rate = 0.01 #nm/ps, rate of change of reference position
        #pull_coord1_k = 1000 # kJ/mol/nm^2, spring constant between referencea nd tracer
        #Mainly just need to specify the pull filename, the coordinates of the reference, 
        #rate of reference moving, and the index groups
         
        #Pulling parameters
        if not os.path.isfile(grofile):
            print('****************')
            print('Error: {} does not exist'.format(grofile))
            print('****************')
            sys.exit()
        pull = 'yes'
        pull_nstxout = '5000'
        pull_nstfout = '5000'
        pull_ncoords = len(tracerlist) #number of coordinates is number of tracers
        pull_ngroups = len(tracerlist) #number of pulling groups is number of tracers

        pull_group_names = list() #Write each groupname
        for i, tracer in enumerate(tracerlist):
            pull_group_names.append(str('Tracer'+str(tracer)))

        pull_coord_groups = list() #Write each pulling group WRT reference
        for i in range(len(pull_group_names)):
                pull_coord_groups.append('0 {}'.format(str(i + 1)))

        #Write index file for each pulling group
        self.write_ndx(grofile, pull_filename, pull_group_names, tracerlist)
        #Default pull parameters, need to apply to each coord
        pull_coord_type = 'umbrella'
        pull_coord_geometry = 'distance'
        pull_coord_dim = 'N N Y'
        pull_coord_start = 'no'

        #print('Writing mdp and ndx files for {}, z_window = {} nm, pull_rate = {} nm/ps'.format(pull_group1_name, np.round(z_window, 2),
        #        pull_coord1_rate))

        #Get the reference coordinates based on tracer coordinate and r_tracer_ref
        r_tracer_ref = 0.1 #(nm)Distance between tracer and reference
        pull_coord_origins = list()
        #Set up origins at a particular z-window
        for i, tracer in enumerate(tracerlist):
                (x,y,z) = self.get_tracer_coordinates(grofile,tracer)
                z = z_window_list[i]
                pull_coord_origins.append((x,y,z))
        
        #Determine t_pulling 
        if not t_pulling:
            if pull_coord_rate == 0:
                #50e3ps = 50ns, this is if we have a fixed reference and we just want to pull the tracer in 
                t_pulling = 50e3
                #so we just let the simulation run for a long time 
            else:
                #In stage 2, we have a moving reference, need to determine the length of time necessary
                #for the reference to hit the z-window
                #This is based on the initial coordinates of the tracer, 
                #desired z-window, and pull_coord_rate. Will also need the grofile to find coordinates
                t_pulling_list = np.ones(len(tracerlist))
                for i, tracer in enumerate(tracerlist):
                    t_pulling_list[i] = self.calc_t_pulling(grofile, tracer, z_window_list[i], pull_coord_rate)
                t_pulling = max(t_pulling_list)
        else:
            t_pulling = t_pulling

        #The following below are MD parameters, less likely to need to change
        #MDP parameters to control
        temp = 305
        pres = 1.0
        #Figure out a pulling time, which depends on initial tracer coordinate and desired tracer window
        #We know the pull rate (nm/ps), or the rate the dummy particle moves
        #Get the coordinate of the dummy particle
        
        #Run MDP parameters
        integrator = 'md'
        dt = 0.002 #ps
        nsteps = int(t_pulling/dt) #nsteps is the time (converted to ps) divided by the step size
        comm_mode = 'Linear' # Remove center of mass translation
        nstcomm = 1 # Remove center of mass motion every step
        comm_grps = 'non-water'
        
        #Output MDP parameters
        nstxout = 0 #Don't save coordinates
        nstvout = 0 #Don't save velocities
        nstxtcout = int(10/dt) # XTC coordinates every 1p0s 
        nstenergy = int(10/dt) #Energy every 10ps
        nstlog = int(10/dt) #Log every 10ps
        if stagefive:
            nstfout = int(1/dt) # Log force every 1 ps
        else: 
            nstfout = 0 #int(1/dt) #Force every 1ps
        
        #Bond parameters 
        continuation = 'yes'
        constraint_algorithm = 'lincs'
        constraints = 'all-bonds'
        lincs_iter = 1
        lincs_order = 4
        
        #Neighbor searching
        cutoff_scheme = 'Verlet'
        nstlist = 10
        rcoulomb = 1.4
        rvdw = 1.4
        
        #Electrostatics
        coulombtype = 'PME'
        fourierspacing = 0.16
        pme_order = 4
        
        #Temperature coupling
        tcoupl = 'nose-hoover'
        tc_grps = '{:8s}\t{:8s}'.format('non-water', 'water')
        tau_t = '{:8s}\t{:8s}'.format('0.4', '0.4')
        ref_t = '{:8s}\t{:8s}'.format(str(temp), str(temp))
        
        #Pressure coupling
        pcoupl = 'Parrinello-Rahman'
        pcoupltype = 'isotropic'
        tau_p = 2.0
        ref_p = pres
        compressibility = 4.5e-5
        refcoord_scaling = 'com'
        
        #Misc stuff
        gen_vel = 'no'
        pbc = 'xyz'
        DispCorr = 'EnerPres'

        #Actually writing the mdp file
        mdpfile = open(pull_filename,'w')
        mdpfile.write('; Run MDP parameters\n')
        if posres:
            mdpfile.write('{:25s} = {}\n'.format('define', ('-D'+posres)))

        mdpfile.write('{:25s} = {}\n'.format('integrator',integrator))
        mdpfile.write('{:25s} = {}\n'.format('dt', str(dt)))
        mdpfile.write('{:25s} = {}\n'.format('nsteps', str(nsteps)))
        mdpfile.write('{:25s} = {}\n'.format('comm-mode', str(comm_mode)))
        mdpfile.write('{:25s} = {}\n'.format('nstcomm', str(nstcomm)))
        mdpfile.write('\n; Output parameters\n')
        mdpfile.write('{:25s} = {}\n'.format('nstxout', str(nstxout)))
        mdpfile.write('{:25s} = {}\n'.format('nstvout', str(nstvout)))
        mdpfile.write('{:25s} = {}\n'.format('nstxtcout', str(nstxtcout)))
        mdpfile.write('{:25s} = {}\n'.format('nstenergy', str(nstenergy)))
        mdpfile.write('{:25s} = {}\n'.format('nstlog', str(nstlog)))
        mdpfile.write('{:25s} = {}\n'.format('nstfout', str(nstfout)))
        mdpfile.write('\n; Bond parameters\n')
        mdpfile.write('{:25s} = {}\n'.format('continuation', str(continuation)))
        mdpfile.write('{:25s} = {}\n'.format('constraint-algorithm', str(constraint_algorithm)))
        mdpfile.write('{:25s} = {}\n'.format('constraints', str(constraints)))
        mdpfile.write('{:25s} = {}\n'.format('lincs-iter', str(lincs_iter)))
        mdpfile.write('{:25s} = {}\n'.format('lincs-order', str(lincs_order)))
        mdpfile.write('\n; Neighbor searching\n') 
        mdpfile.write('{:25s} = {}\n'.format('cutoff-scheme', str(cutoff_scheme)))
        mdpfile.write('{:25s} = {}\n'.format('nstlist', str(nstlist)))
        mdpfile.write('{:25s} = {}\n'.format('rcoulomb', str(rcoulomb)))
        mdpfile.write('{:25s} = {}\n'.format('rvdw', str(rvdw)))
        mdpfile.write('\n; Electrostatics\n')
        mdpfile.write('{:25s} = {}\n'.format('coulombtype', str(coulombtype)))
        mdpfile.write('{:25s} = {}\n'.format('fourierspacing', str(fourierspacing)))
        mdpfile.write('{:25s} = {}\n'.format('pme_order', str(pme_order)))
        mdpfile.write('\n; Temperature coupling\n')
        mdpfile.write('{:25s} = {}\n'.format('tcoupl', str(tcoupl)))
        mdpfile.write('{:25s} = {}\n'.format('tc_grps', str(tc_grps)))
        mdpfile.write('{:25s} = {}\n'.format('tau_t', str(tau_t)))
        mdpfile.write('{:25s} = {}\n'.format('ref_t', str(ref_t)))
        mdpfile.write('\n; Pressure coupling\n')
        mdpfile.write('{:25s} = {}\n'.format('pcoupl', str(pcoupl)))
        mdpfile.write('{:25s} = {}\n'.format('pcoupltype', str(pcoupltype)))
        mdpfile.write('{:25s} = {}\n'.format('tau_p', str(tau_p)))
        mdpfile.write('{:25s} = {}\n'.format('ref_p', str(ref_p)))
        mdpfile.write('{:25s} = {}\n'.format('compressibility', str(compressibility)))
        mdpfile.write('{:25s} = {}\n'.format('refcoord_scaling', str(refcoord_scaling)))
        mdpfile.write('\n; Misc stuff\n')
        mdpfile.write('{:25s} = {}\n'.format('gen_vel', str(gen_vel)))
        mdpfile.write('{:25s} = {}\n'.format('pbc', str(pbc)))
        mdpfile.write('{:25s} = {}\n'.format('DispCorr', str(DispCorr)))
        if not posres:
            mdpfile.write('\n; Pull parameters\n')
            mdpfile.write('{:25s} = {}\n'.format('pull', str(pull)))
            mdpfile.write('{:25s} = {}\n'.format('pull-nstxout', str(pull_nstxout)))
            mdpfile.write('{:25s} = {}\n'.format('pull-nstfout', str(pull_nstfout)))
            mdpfile.write('{:25s} = {}\n'.format('pull-ngroups', str(pull_ngroups)))
            mdpfile.write('{:25s} = {}\n'.format('pull-ncoords', str(pull_ncoords)))
            for i in range(len(tracerlist)):
                mdpfile.write('{:25s} = {}\n'.format('pull-group'+str(i+1)+'-name', pull_group_names[i]))        
                #mdpfile.write('{:25s} = {}\n'.format('pull_coord'+str(i+1)+'_name', pull_coord_type))
                mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-groups', pull_coord_groups[i]))
                mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-type', pull_coord_type))
                mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-geometry', pull_coord_geometry))
                mdpfile.write('{:25s} = {:<8.3f} {:<8.3f} {:<8.3f}\n'.format('pull-coord'+str(i+1)+'-origin', 
                    pull_coord_origins[i][0], pull_coord_origins[i][1], pull_coord_origins[i][2]))
                mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-dim', pull_coord_dim))
                mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-rate', pull_coord_rate))
                mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-k', pull_coord_k))
                mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-start', pull_coord_start))
            #mdpfile.write('{:25s} = {}\n'.format('pull_group1_name', str(pull_group1_name)))
            #mdpfile.write('{:25s} = {}\n'.format('pull_coord1_type', str(pull_coord1_type)))
            #mdpfile.write('{:25s} = {}\n'.format('pull_coord1_groups', str(pull_coord1_groups)))
            #mdpfile.write('{:25s} = {:<8.3f} {:<8.3f} {:<8.3f}\n'.format('pull_coord1_origin', pull_coord1_origin[0], pull_coord1_origin[1],
            #    pull_coord1_origin[2]))
            #mdpfile.write('{:25s} = {}\n'.format('pull_coord1_dim', str(pull_coord1_dim)))
            #mdpfile.write('{:25s} = {}\t ;nm/ps\n'.format('pull_coord1_rate', str(pull_coord1_rate)))
            #mdpfile.write('{:25s} = {}\t ;kJ/mol/nm^2\n'.format('pull_coord1_k', str(pull_coord1_k)))
            #mdpfile.write('{:25s} = {}\n'.format('pull_coord1_start', str(pull_coord1_start)))
        mdpfile.close()

    def calc_t_pulling(self, grofile, tracer, z_window, pull_coord_rate):
        #Calculate the time necessary for the reference to move into a particular z-window
        #Gather the coordinates of the tracer
        (x_ref, y_ref, z_ref) = self.get_tracer_coordinates(grofile, tracer)

        distance_to_traverse = np.abs(float(z_window) - float(z_ref)) #nm
        t_pulling = distance_to_traverse / pull_coord_rate #ps

        return t_pulling

    def get_tracer_coordinates(self, grofile, tracer):
        #Read the .gro file and get the coordinates of the tracer
        #Well, we need to get the cetner of mass, so maybe the oxygen's coordinate, which is the first to appear
        filename = open(str(grofile),'r')
        grolines = filename.readlines()
        
        (x, y, z) = (0, 0, 0)
        #Find the tracer molecule in the list
        for i, line in enumerate(grolines):
            stuff = line.split()
            if stuff[0] == (str(tracer)+'SOL'):
                #Assumes grofile also has velocities in addition ot positions
                (x,y,z) = (float(stuff[-6]), float(stuff[-5]), float(stuff[-4])) 
                break
            else:
                pass

        return (x,y,z)
    
    def write_ndx(self, grofile, pull_filename, pull_group_names, tracerlist):
        #Write index file for each pulling group
        #Read grofile and obtain indices for each atom relevant to the given tracer
        #Find the index of the oxygen atom, then the next two indices are the hydrogen

        #Loop through grofile to find the line with oxygen
        filename = open(str(grofile),'r')
        grolines = filename.readlines()
        #print(grolines)
        targetlines = [None] * len(tracerlist)
        #loop through gro file
        for i, line in enumerate(grolines):
            stuff = line.split()
            #print(line)
            #For each line, check to see if it contains the tracer in question
            for j, tracer in enumerate(tracerlist):
                if stuff[0] == (str(tracer)+'SOL'):
                    #Check if the targetline for the given tracer was already found (we want the first occurrence of the molecule)
                    if targetlines[j] is None:
                        targetlines[j] = line
                    else:
                        pass
                else:
                    pass

        oxygenindices = [None] * len(tracerlist) 
        if targetlines[0] == None:
            print('************************')
            print('Error identifying tracer lines in grofile')
            print('************************')
            sys.exit()
        #First loop through targetlines to get each targetline
        #Loop through targetline to find atom index of the oxygen
        for j, targetline in enumerate(targetlines):
            for i,c in enumerate(targetline):
                if str(targetline[i]+targetline[i+1]) == "OW":
                    oxygenindex = int(targetline[i+2:i+9])
                    oxygenindices[j] = oxygenindex
                    break
                else:
                    pass
        outfile = open(str(pull_filename[:-4] + '.ndx'), 'w')

        #Loop through tracer names and oxygen indices
        for i, groupname in enumerate(pull_group_names):
            outfile.write('[ {} ]\n'.format(groupname))
            outfile.write('{:10.0f}\t{:10.0f}\t{:10.0f}\n'.format(
                float(oxygenindices[i]), float(oxygenindices[i])+1, float(oxygenindices[i])+2))

    def write_tracerlist(self, tracer_list, tracerlog = 'tracers.out'):
        outfile = open(tracerlog, 'w')
        for i, tracer in enumerate(tracer_list):
            outfile.write('{} \n'.format(tracer))
        outfile.close()

    def write_zlist(self, z_list, zlog = 'z_windows.out'):
        outfile = open(zlog, 'w')
        for i, zwindow in enumerate(z_list):
            outfile.write('{} \n'.format(np.round(zwindow,2)))
        outfile.close()

    def write_pbs(self, pbsname, grofile, topfile, directoryname1, mdpfile1, 
            directoryname2, mdpfile2):
        #Due to the nature of open MP and gpu threads, this pbs script will write a submission script
        # to run two simulations, allotting 8 open mp threads per gpu (there are two gpu)
        pbsflags = ('#PBS -N {}\n'.format(pbsname)+
                '#PBS -q batch\n'+
                '#PBS -l nodes=1, walltime=02:00:00\n'+
                '#PBS -j oe\n' + 
                '#PBS -A BIP140\n'+ 
                '#PBS -m ae\n'+
                '#PBS -V\n')
            
        #pbsflags = ('#PBS -N {} \n#PBS -l nodes=1:ppn=16 \n#PBS -l walltime=96:00:00 \n#PBS -q low'.format((pbsname))+
        #        '\n#PBS -m abe \n#PBS -M {}\n\n'.format('alexander.h.yang@vanderbilt.edu'))
        pbsflags2 = ('cd $PBS_O_WORKDIR \necho `cat $PBS_NODEFILE`\n\nexport CRAY_CUDA_MPS=1\n')
        module_load = 'module load gromacs/5.1.0\n'

        tprfile1 = mdpfile1[:-4] + '.tpr'
        tprfile2 = mdpfile2[:-4] + '.tpr'

        ndxfile1 = mdpfile1[:-4] + '.ndx'
        ndxfile2 = mdpfile2[:-4] + '.ndx'

        forcefile1 = mdpfile1[:-4] + '_pullf.xvg'
        comfile1 = mdpfile1[:-4] + '_pullx.xvg'

        forcefile2 = mdpfile2[:-4] + '_pullf.xvg'
        comfile2 = mdpfile2[:-4] + '_pullx.xvg'


        grompp_line1 = 'gmx grompp -f {} -p {} -c {} -n {} -o {}\n'.format(mdpfile1,
                topfile, grofile, (directoryname1 + '/' + 'FullIndex.ndx'), tprfile1)
        grompp_line2 = 'gmx grompp -f {} -p {} -c {} -n {} -o {}\n'.format(mdpfile2,
                topfile, grofile, (directoryname2 + '/' + 'FullIndex.ndx'), tprfile2)

        mdrun_line1 = 'aprun -n 8 -N 8 gmx_mpi mdrun -ntomp 1 -gpu_id 00000000 -deffnm {}\n'.format( mdpfile1)
        mdrun_line2 = 'aprun -n 8 -N 8 gmx_mpi mdrun -ntomp 1 -gpu_id 00000000 -deffnm {}\n'.format( mdpfile2)

        cptfile1 = (tprfile1[:-4] + '.cpt')
        cptfile2 = (tprfile2[:-4] + '.cpt')

        cont_line1 = ('aprun -n 8 -N 8 gmx_mpi mdrun -ntomp 1 -gpu_id 00000000 -append -s {} -cpi {} -deffnm' +
            '{}\n-px {}\n-pf {}'.format(tprfile1, cptfile1, tprfile1[:-4]),comfile1, forcefile1)
        cont_line2 = ('aprun -n 8 -N 8 gmx_mpi mdrun -ntomp 1 -gpu_id 00000000 -append -s {} -cpi {} -deffnm'+
            '{}\n-px {}\n-pf {}'.format(tprfile2, cptfile2, tprfile2[:-4]),comfile2, forcefile2)


        outfile = open((pbsname)+'_start.pbs','w')
        outfile.write(pbsflags)
        outfile.write(pbsflags2)
        outfile.write(module_load)
        #outfile.write(grompp_line1)
        #outfile.write(grompp_line2)
        outfile.write(mdrun_line1)
        outfile.write(mdrun_line2)
        outfile.close()

        continuefile = open((pbsname)+'_cont.pbs','w')
        continuefile.write(pbsflags)
        continuefile.write(pbsflags2)
        continuefile.write(module_load)
        continuefile.write(cont_line1)
        continuefile.write(cont_line2)
        continuefile.close()


    def write_pbs_single(self, pbsname, grofile, topfile, directoryname1, mdpfile1): 
        #Due to the nature of open MP and gpu threads, this pbs script will write a submission script
        # to run two simulations, allotting 8 open mp threads per gpu (there are two gpu)
        #But this is single if this a sole simulation to run

        pbsflags = ('#PBS -N {}\n'.format(pbsname)+
                '#PBS -q batch\n'+
                '#PBS -l nodes=1, walltime=02:00:00\n'+
                '#PBS -j oe\n' + 
                '#PBS -A BIP140\n'+ 
                '#PBS -m ae\n'+
                '#PBS -V\n')

        #pbsflags = ('#PBS -N {} \n#PBS -l nodes=1:ppn=16 \n#PBS -l walltime=96:00:00 \n#PBS -q low'.format((pbsname))+
        #        '\n#PBS -m abe \n#PBS -M {}\n\n'.format('alexander.h.yang@vanderbilt.edu'))
        #pbsflags2 = ('cd $PBS_O_WORKDIR \necho `cat $PBS_NODEFILE`\n\n')
        pbsflags2 = ('cd $PBS_O_WORKDIR \necho `cat $PBS_NODEFILE`\n\nexport CRAY_CUDA_MPS=1\n')

        module_load = 'module load gromacs/5.1.0\n'

        tprfile1 = mdpfile1[:-4] + '.tpr'

        ndxfile1 = mdpfile1[:-4] + '.ndx'

        forcefile1 = mdpfile1[:-4] + '_pullf.xvg'
        comfile1 = mdpfile1[:-4] + '_pullx.xvg'


        grompp_line1 = 'gmx grompp -f {} -p {} -c {} -n {} -o {}\n'.format(mdpfile1,
                topfile, grofile, (str(directoryname1) + 
                '/' + 'FullIndex.ndx'), tprfile1)

        mdrun_line1 = 'aprun -n 8 -N 8 gmx_mpi mdrun -ntomp 8 -gpu_id 00000000 -deffnm {}\n'.format( mdpfile1)

        cptfile1 = (tprfile1[:-4] + '.cpt')

        cont_line1 = ('aprun -n 8 -N 8 gmx_mpi mdrun -ntomp 8 -gpu_id 00000000 -append -s {} -cpi {} -deffnm'+
            '{}\n-pf {}\n-px {}\n'.format(tprfile1, cptfile1, tprfile1[:-4], forcefile1, comfile1))


        outfile = open((pbsname)+'_start.pbs','w')
        outfile.write(pbsflags)
        outfile.write(pbsflags2)
        outfile.write(module_load)
        #outfile.write(grompp_line1)
        outfile.write(mdrun_line1)
        outfile.close()

        continuefile = open((pbsname)+'_cont.pbs','w')
        continuefile.write(pbsflags)
        continuefile.write(pbsflags2)
        continuefile.write(module_load)
        continuefile.write(cont_line1)
        continuefile.close()

    def write_slurm(self, directoryname, slurmname, mdpfile, grofile, topfile):
        slurmflags = ('#SBATCH -p regular \n#SBATCH -N 1 \n#SBATCH -t 08:00:00 \n#SBATCH -J {} \n'.format((slurmname))+
                '#SBATCH -o {} \n#SBATCH --mail-type=ALL \n#SBATCH --mail-user={}\n\n'.format(
                    slurmname, 'alexander.h.yang@vanderbilt.edu'))
        module_load = 'module load gromacs/5.1.2\n'
        grompp_line = 'gmx_sp grompp -f {} -p {} -c {} -n {} -o {}\n'.format(mdpfile, topfile, grofile, (directoryname + '/' +
            'FullIndex.ndx'),(slurmname+'.tpr'))
        mdrun_line = 'srun -n 12 mdrun_mpi_sp -ntmpi 12 -deffnm {}\n'.format(slurmname)

        cont_line = 'srun -n 12 mdrun_mpi_sp -ntmpi 12 -append -s {} -cpi {}\n'.format((slurmname+'.tpr'), (slurmname+'.cpt'))


        outfile = open((directoryname + '/' + slurmname)+'_start.sbatch','w')
        outfile.write(slurmflags)
        outfile.write(module_load)
        #outfile.write(grompp_line)
        outfile.write(mdrun_line)
        outfile.close()

        continuefile = open((directoryname + '/' + slurmname)+'_cont.sbatch','w')
        continuefile.write(slurmflags)
        continuefile.write(module_load)
        continuefile.write(cont_line)
        continuefile.close()
      
    def write_slurm_stage2(self, directoryname, slurmname, mdpfile, grofile, oldfilename):
        slurmflags = ('#SBATCH -p regular \n#SBATCH -N 1 \n#SBATCH -t 08:00:00 \n#SBATCH -J {} \n'.format((slurmname))+
                '#SBATCH -o {} \n#SBATCH --mail-type=ALL \n#SBATCH --mail-user={}\n\n'.format(
                    slurmname, 'alexander.h.yang@vanderbilt.edu'))
        module_load = 'module load gromacs/5.1.2\n'
        grompp_line = 'gmx_sp grompp -f {} -n {} -c {} -o'.format(mdpfile, (directoryname + '/' + 'FullIndex.ndx') ,(oldfilename + '.tpr'), (slurmname + '.tpr'))
        mdrun_line = 'srun -n 12 mdrun_mpi_sp -ntmpi 12 -cpi {} -deffnm {}\n'.format((oldfilename + '.cpt'), slurmname)
        cont_line = 'srun -n 12 mdrun_mpi_sp -ntmpi 12 -append -s {} -cpi {}\n'.format((slurmname+'.tpr'), (slurmname+'.cpt'))

        outfile = open((directoryname + '/' + slurmname)+'_start.sbatch','w')
        outfile.write(slurmflags)
        outfile.write(module_load)
        #outfile.write(grompp_line)
        outfile.write(mdrun_line)
        outfile.close()

        continuefile = open((directoryname + '/' + slurmname)+'_cont.sbatch','w')
        continuefile.write(slurmflags)
        continuefile.write(module_load)
        continuefile.write(cont_line)
        continuefile.close()

    def write_pbs_stage2(self, pbsname, oldfilename1, directoryname1, mdpfile1, 
            oldfilename2, directoryname2, mdpfile2):
        #Due to the nature of open MP and gpu threads, this pbs script will write a submission script
        # to run two simulations, allotting 8 open mp threads per gpu (there are two gpu)
        pbsflags = ('#PBS -N {} \n#PBS -l nodes=1:ppn=16 \n#PBS -l walltime=96:00:00 \n#PBS -q low'.format((pbsname))+
                '\n#PBS -m abe \n#PBS -M {}\n\n'.format('alexander.h.yang@vanderbilt.edu'))
        pbsflags2 = ('cd $PBS_O_WORKDIR \necho `cat $PBS_NODEFILE`\n\n')
        module_load = 'module load gromacs/5.1.4\n'

        tprfile1 = mdpfile1[:-4] + '.tpr'
        tprfile2 = mdpfile2[:-4] + '.tpr'

        ndxfile1 = oldfilename1 + '.ndx'
        ndxfile2 = oldfilename2 + '.ndx'


        grompp_line1 = 'gmx grompp -f {} -c {} -n {} -o {}\n'.format(mdpfile1,
                (oldfilename1 + '.tpr') , (directoryname1 + '/' + 'FullIndex.ndx'), tprfile1)
        grompp_line2 = 'gmx grompp -f {} -c {} -n {} -o {}\n'.format(mdpfile2,
                (oldfilename2 + '.tpr') , (directoryname2 + '/' + 'FullIndex.ndx'), tprfile2)

        mdrun_line1 = 'gmx mdrun -ntomp 8 -gpu_id 0 -cpi -deffnm {}\n'.format( (oldfilename1 + '.cpt'), mdpfile1[:-4])
        mdrun_line2 = 'gmx mdrun -ntomp 8 -gpu_id 1 -cpi -deffnm {}\n'.format( (oldfilename2 + '.cpt'), mdpfile2[:-4])

        cptfile1 = (tprfile1[:-4] + '.cpt')
        cptfile2 = (tprfile2[:-4] + '.cpt')

        cont_line1 = ('gmx mdrun -ntomp 8 -gpu_id 0 -append -s {} -cpi {} -deffnm'+
            '{}\n'.format(tprfile1, cptfile1, tprfile1[:-4]))
        cont_line2 = ('gmx mdrun -ntomp 8 -gpu_id 1 -append -s {} -cpi {} -deffnm'+
            '{}\n'.format(tprfile2, cptfile2, tprfile2[:-4]))


        outfile = open((pbsname)+'_start.pbs','w')
        outfile.write(pbsflags)
        outfile.write(pbsflags2)
        outfile.write(module_load)
        #outfile.write(grompp_line1)
        #outfile.write(grompp_line2)
        outfile.write(mdrun_line1)
        outfile.write(mdrun_line2)
        outfile.close()

        continuefile = open((pbsname)+'_cont.pbs','w')
        continuefile.write(pbsflags)
        continuefile.write(pbsflags2)
        continuefile.write(module_load)
        continuefile.write(cont_line1)
        continuefile.write(cont_line2)
        continuefile.close()


    def write_pbs_single_stage2(self, pbsname, oldfilename1,  directoryname1, mdpfile1): 
        #Due to the nature of open MP and gpu threads, this pbs script will write a submission script
        # to run two simulations, allotting 8 open mp threads per gpu (there are two gpu)
        #But this is single if this a sole simulation to run
        pbsflags = ('#PBS -N {} \n#PBS -l nodes=1:ppn=16 \n#PBS -l walltime=96:00:00 \n#PBS -q low'.format((pbsname))+
                '\n#PBS -m abe \n#PBS -M {}\n\n'.format('alexander.h.yang@vanderbilt.edu'))
        pbsflags2 = ('cd $PBS_O_WORKDIR \necho `cat $PBS_NODEFILE`\n\n')
        module_load = 'module load gromacs/5.1.4\n'

        tprfile1 = mdpfile1[:-4] + '.tpr'

        ndxfile1 = mdpfile1[:-4] + '.ndx'

        oldtpr1 = oldfilename1 + '.tpr'

        oldcpt1 = oldfilename1 + '.cpt'

        grompp_line1 = 'gmx grompp -f {} -c {} -n {} -o {}\n'.format(mdpfile1,
                oldtpr1, (directoryname1 + '/' + 'FullIndex.ndx'), tprfile1)

        mdrun_line1 = 'gmx mdrun -ntomp 8 -gpu_id 0 -cpt -deffnm {}\n'.format((tprfile1[:-4] + '.cpt'), mdpfile1)

        cptfile1 = (tprfile1[:-4] + '.cpt')

        cont_line1 = ('gmx mdrun -ntomp 8 -gpu_id 0 -append -s {} -cpi {} -deffnm'+
            '{}\n'.format(tprfile1, cptfile1, tprfile1[:-4]))


        outfile = open((pbsname)+'_start.pbs','w')
        outfile.write(pbsflags)
        outfile.write(pbsflags2)
        outfile.write(module_load)
        #outfile.write(grompp_line1)
        outfile.write(mdrun_line1)
        outfile.close()

        continuefile = open((pbsname)+'_cont.pbs','w')
        continuefile.write(pbsflags)
        continuefile.write(pbsflags2)
        continuefile.write(module_load)
        continuefile.write(cont_line1)
        continuefile.close()

    def write_grompp_file(self, directoryname, filename, grofile, mdpfile, indexfile, topfile = None, oldtpr = None, cptfile = None, posres
            = None): 
        gromppfilename = '{}/Grompp_{}.sh'.format(directoryname, filename)
        outfile = open(gromppfilename, 'w')
#        if oldtpr and cptfile:
#        #IF building off an exisitng tpr and trajectory
#            outfile.write('gmx grompp -f {} -p {} -n {} -c {} -t {} -o {} -maxwarn 2 > grompp_{}.log 2>&1'.format((mdpfile), (topfile), (indexfile),
#               (oldtpr), (cptfile), (filename + '.tpr'), filename))
#
#        #If building a grompp mdp from scratch
#        elif topfile:
        outfile.write('gmx grompp -f {} -c {} -p {} -n {} -o {} -maxwarn 2 > grompp_{}.log 2>&1'.format((mdpfile), (grofile),
              topfile, (indexfile), (filename+'.tpr'), filename))


        outfile.close()

    def write_posres(self, directoryname, posres, tracer_list, grofile):
        
        #Fix to the X-Y plane
        #Write the position restraint file
        posresfile = open((str(directoryname) + '/'+ str(posres)+'.itp'),'w')
        posresfile.write('[ position_restraints ]\n')
        posresfile.write(';{:10s}{:10s}{:10s}{:10s}{:10s}\n'.format('ai', 'funct', 'fcx', 'fcy', 'fcz'))
        #for i, atomnumber in enumerate(fixatom_list):
        for i in range(1,4):
            posresfile.write('{:>10.0f}{:>10.0f}{:>10.0f}{:>10.0f}{:>10.0f}\n'.format(i, 1, 0, 0, 1000))
        posresfile.close()

        
    def add_posres(self, directoryname, topfile, posres, grofile, tracer_list):
        '''
        Add position restraint to the topology file
        Also means repacing certain lines in the grofile
        This includes modifying the topology, replacing some SOL molecules with Tracer molecules
        And then adding the position restraing
        '''
        #Grofile editing
        inputgrofilename = open( str(grofile), 'r')
        inputgrolines = inputgrofilename.readlines()
        #Make a copy of the grofile lines, outlines, that we end up printing
        outputgrolines = inputgrolines 
        atomcounter = 0
        #Loop through gro file, getting atom indices of the relevant tracers
        for i, line in enumerate(inputgrolines):
            splitline = line.split()
            #Keep count of which atom we're on based on number of elements in the groline row
            if len(splitline)  >= 4:
                atomcounter+=1
            #For a given line or atom, check to see if it belongs to the tracer
            #Based on it's tracer number (molecule)
            for j, tracernumber in enumerate(tracer_list):
                if splitline[0] == (str(tracernumber)+'HOH'):
                    #If we've hit a molecule in the grofile correspnoding to the tracer
                    # Rewrite the line to be a trace molecule
                    outputgrolines[i] = ('{:>5s}{:5s}'.format(tracernumber, 'Trace') + line[10:-1]+'\n')

        inputgrofilename.close()
        #Print outputlines to the grofile    
        outputfile = open( str(grofile),'w')
        for i, line in enumerate(outputgrolines):
            outputfile.write(line)
        outputfile.close()


        infile = open( str(directoryname+'/'+topfile), 'r')
        toplines = infile.readlines()
        infile.close()

        #Read in lines, add the include trace.itp and posres lines,after the include spc line
        outfile = open( str(directoryname + '/' + topfile), 'w')
        include_spc = False
        molecules_directive = False
        for i, line in enumerate(toplines):
            if molecules_directive:
                if 'SOL' in line:
                    #Rewrite SOL line by subrtracting tracer molecules
                    #Then write the Tracer line
                    n_sol = int(line.split()[1])
                    #print (str(n_sol))
                    new_n_sol = n_sol - len(tracer_list)
                    outfile.write('{:<10s} {:<10.0f}\n'.format(line.split()[0], new_n_sol))
                    outfile.write('{:<10s} {:<10.0f}\n'.format('Trace', len(tracer_list)))
                else:
                    outfile.write(line)
            else: 
                outfile.write(line)
                if 'spc.itp' in line:
                    include_spc = True
                    # If you've hit the include spc.itp line, start printing the include trace.itp line
                    # And the define posres line
                if include_spc:
                    outfile.write('; Include Tracer water topology \n')
                    outfile.write('#include \"/raid6/homes/ahy3nz/Programs/setup/FF/gromos53a6/trace.itp\" \n')
                    outfile.write('#ifdef POSRES\n')
                    outfile.write('#include "{}.itp"\n'.format(posres))
                    outfile.write('#endif\n')
                    include_spc = False
                if '[ molecules ]' in line:
                    molecules_directive = True
                    # If you've hit the molecules directive, start printing the new system
                
        #Modify the [ molecules ] to accoutn for new SOL and new Trace
        outfile.close()
