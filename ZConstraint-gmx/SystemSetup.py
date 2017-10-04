import numpy as np
import mdtraj
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
    def __init__(self, z0=1.0, dz=0.2, N_window=40, N_tracer=8, 
            z_windows=None, auto_detect=False, grofile=None):
        """

        Parameters
        ----------
        z0 : float
            Initial z coordinate
        dz : float
            window spacing
        N_window : int
            Number of z-windows
        N_tracer : int 
            Number of tracer molecules
        Z_windows : str
            Filename of z-windows if specified
        auto_detect : Boolean
            If true, automatically generate z windows based on bilayer CoM
        grofile : string
            Filename of gmx structure file
            """
        # If no z window file was provied
            
        self._N_tracer = N_tracer
        self._grofile = grofile
        
        if auto_detect:
            print("Generating windows from center of mass")
            self._generate_z_windows(grofile=grofile, dz=dz, N_window=N_window)
            self.write_zlist()

        elif z_windows is None:
            print("Generating windows from window specifications")
            self._z0 = z0
            self._dz = dz
            self._N_window = N_window
       
            #Define the z-window list, starting at z0 and going up dz each time
        
            self._zlist = list()
            for i in range(self._N_window):
                   self._zlist.append(self._z0 + (i * self._dz)) 
            self.write_zlist()
        elif z_windows:
            print("Loading window from file")
            # IF we have a z window file, just set it
            self._set_from_file(z_windows)
        else:
            sys.exit("Specify SystemSetup initializaion parameters!")

    def _generate_z_windows(self, grofile=None, dz=0.2, N_window=40):
        """ Build zwindows out from the center of mass"""

        traj = mdtraj.load(grofile)
        non_water = traj.topology.select('not water')
        sub_traj = traj.atom_slice(non_water)
        com = mdtraj.compute_center_of_mass(sub_traj)
        center_z = round(com[0,2], 2)
        self._zlist = np.zeros(N_window)
        midpoint = int(len(self._zlist)/2)
        self._zlist[midpoint] = center_z

        # Fill in lower half from centerpoint outward
        for i in range(1, midpoint+1):
            self._zlist[midpoint-i] = center_z - i*dz
        # Fill in top half from centerpoint outward
        for i in range(midpoint+1, len(self._zlist)):
            self._zlist[i] = self._zlist[0] + i*dz
        




    def _set_from_file(self, z_windows):
        with open(z_windows) as f:
            z_lines = f.readlines()
        self._zlist = [line.strip() for line in z_lines]
        self._z0 = round(float(self._zlist[0]), 2)
        self._dz = round(float(self._zlist[1]) - float(self._zlist[0]), 2)
        self._N_window = len(self._zlist)


    def write_zlist(self, zlog='z_windows.out'):
        outfile = open(zlog, 'w')
        for i, zwindow in enumerate(self._zlist):
            outfile.write('{} \n'.format(np.round(zwindow,2)))
        outfile.close()

    @property
    def dz(self):
        return round(float(self._dz), 2)

    @property
    def z0(self):
        return round(float(self._z0), 2)



    @property
    def tracer_list(self):
        return self._tracer_list

    @tracer_list.setter
    def tracer_list(self, tracerfile):
        with open(tracerfile) as f:
            tracer_lines = f.readlines()
        self._tracer_list = [line.strip() for line in tracer_lines]

    def get_Tracers(self):
        return self.tracer_list

    @property
    def zlist(self):
        return self._zlist

    def get_zlist(self):
        return self._zlist

    @zlist.setter
    def zlist(self, window_file):
        with open(window_file) as f:
            z_lines = f.readlines()
        self._zlist = [line.strip() for line in z_lines]

    
    def get_z0(self):
        return self.z0

    

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
        """
        Parameters
        ---------
        tracer_list : str
            filename of tracers
            """
        #self._tracer_list = list()
        #for i, tracer in enumerate(tracer_list):
            #self._tracer_list.append(tracer.split()[0])
        self._tracer_list = np.loadtxt(tracer_list,dypte=int)

    def read_zlist(self, z_list):
        """
        Parameters
        ---------
        z_list : str
            filename of zwindows
            """
        #self.zlist = list()
        self._zlist = np.loadtxt(z_list)
        #for i, zwindow in enumerate(z_list):
            #self._zlist.append(zwindow.split()[0])
        self._dz = np.absolute(float(self._zlist[0]) - float(self._zlist[1]))
        self._z0 = self._zlist[0]

    def write_pulling_mdp(self, pull_filename=None, tracerlist=None, 
            z_window_list=None, 
            grofile=None, pull_coord_rate=0,pull_coord_k=1000,  
            t_pulling=None, stagefive=False, stagethree = False, 
            moving_sim = False): 
        #Set up a pulling force on each tracer at very far windows
        #specify the group
        #Pulling reaction coordinate is just one direction (z)
        #Specify the pulling groups, 0 is a reference and 1 is the tracer
        #Set the reference position, then we will need to eliminate center of mass motion
        #pull_coord1_rate = 0.01 #nm/ps, rate of change of reference position
        #pull_coord1_k = 1000 # kJ/mol/nm^2, spring constant between referencea nd tracer
        #Mainly just need to specify the pull filename, the coordinates of the reference, 
        #rate of reference moving, and the index groups

        #Run MDP parameters
        integrator = 'md'
        dt = 0.002 #ps
        t_pulling = 1e3
        nsteps = int(t_pulling/dt) #nsteps is the time (converted to ps) divided by the step size
        comm_mode = 'Linear' # Remove center of mass translation
        nstcomm = 1 # Remove center of mass motion every step
        comm_grps = 'non-water water'
         
        #Pulling parameters
        if not os.path.isfile(grofile):
            print('****************')
            print('Error: {} does not exist'.format(grofile))
            print('****************')
            sys.exit()
        pull = 'yes'
        if stagefive:
            pull_nstxout = int(0.01/dt) # log forces every 0.01 ps
            pull_nstfout = int(0.01/dt) # log forces every 0.01 ps
        else:
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
        #pull_coord_geometry = 'distance'
        if stagethree:
            pull_coord_geometry = 'direction-periodic'
        else:
            pull_coord_geometry = 'direction'
        pull_coord_dim = 'N N Y'
        pull_coord_vec = '0 0 1'
        pull_coord_start = 'no'


        #Get the reference coordinates based on tracer coordinate and r_tracer_ref
        r_tracer_ref = 0.1 #(nm)Distance between tracer and reference
        pull_coord_origins = list()
        #Set up origins at a particular z-window
        if moving_sim:
            for i, tracer in enumerate(tracerlist):
                (x,y,z) = self.get_tracer_coordinates(grofile, tracer)
                z += r_tracer_ref
                pull_coord_origins.append((x,y,z))

        else:
            for i, tracer in enumerate(tracerlist):
                    (x,y,z) = self.get_tracer_coordinates(grofile,tracer)
                    z = z_window_list[i]
                    pull_coord_origins.append((x,y,z))
        
        pull_coord_rate_list = pull_coord_rate*np.ones(len(tracerlist))
        #Determine t_pulling 
        # Determine how fast the pulling reference moves
        if moving_sim:
            t_pulling = 1e3
            for i, tracer in enumerate(tracerlist):
                pull_coord_rate_list[i] = self.calc_pulling_rate(grofile, tracer, z_window_list[i], t_pulling)


        #The following below are MD parameters, less likely to need to change
        #MDP parameters to control
        temp = 305
        pres = 1.0
        #Figure out a pulling time, which depends on initial tracer coordinate and desired tracer window
        #We know the pull rate (nm/ps), or the rate the dummy particle moves
        #Get the coordinate of the dummy particle
        
        
        
        #Output MDP parameters
        nstxout = 0 #Don't save coordinates
        nstvout = 0 #Don't save velocities
        nstxtcout = int(10/dt) # XTC coordinates every 1p0s 
        nstenergy = int(10/dt) #Energy every 10ps
        nstlog = int(10/dt) #Log every 10ps
        nstcalcenergy = 1
        nstfout = 0  # No force logging
        if stagefive:
            nstxout = int(10/dt)
            nstvout = int(10/dt)
            nstfout = int(10/dt)
        
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
        if stagethree:
            pcoupl = 'no'
            pcoupltype = ''
            tau_p = 0.0
            ref_p = '0 0'
            compressibility = '0 0'
            refcoord_scaling = 'com'
        else:
            pcoupl = 'Parrinello-Rahman'
            pcoupltype = 'semiisotropic'
            tau_p = 2.0
            ref_p = '{} {}'.format(pres,pres)
            compressibility = '4.5e-5 4.5e-5'
            refcoord_scaling = 'com'
        
        #Misc stuff
        gen_vel = 'no'
        pbc = 'xyz'
        DispCorr = 'EnerPres'

        #Actually writing the mdp file
        mdpfile = open(pull_filename,'w')
        mdpfile.write('; Run MDP parameters\n')

        mdpfile.write('{:25s} = {}\n'.format('integrator',integrator))
        mdpfile.write('{:25s} = {}\n'.format('dt', str(dt)))
        mdpfile.write('{:25s} = {}\n'.format('nsteps', str(nsteps)))
        mdpfile.write('{:25s} = {}\n'.format('comm-mode', str(comm_mode)))
        mdpfile.write('{:25s} = {}\n'.format('nstcomm', str(nstcomm)))
        mdpfile.write('{:25s} = {}\n'.format('comm-grps', str(comm_grps)))
        mdpfile.write('\n; Output parameters\n')
        mdpfile.write('{:25s} = {}\n'.format('nstxout', str(nstxout)))
        mdpfile.write('{:25s} = {}\n'.format('nstvout', str(nstvout)))
        mdpfile.write('{:25s} = {}\n'.format('nstxtcout', str(nstxtcout)))
        mdpfile.write('{:25s} = {}\n'.format('nstenergy', str(nstenergy)))
        mdpfile.write('{:25s} = {}\n'.format('nstlog', str(nstlog)))
        mdpfile.write('{:25s} = {}\n'.format('nstfout', str(nstfout)))
        mdpfile.write('{:25s} = {}\n'.format('nstcalcenergy', str(nstcalcenergy)))
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
            mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-vec', pull_coord_vec))
            mdpfile.write('{:25s} = {:<8.3f} {:<8.3f} {:<8.3f}\n'.format('pull-coord'+str(i+1)+'-origin', 
                pull_coord_origins[i][0], pull_coord_origins[i][1], pull_coord_origins[i][2]))
            mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-dim', pull_coord_dim))
            mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-rate', pull_coord_rate_list[i]))
            mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-k', pull_coord_k))
            mdpfile.write('{:25s} = {}\n'.format('pull-coord'+str(i+1)+'-start', pull_coord_start))
        mdpfile.close()

    def calc_t_pulling(self, grofile, tracer, z_window, pull_coord_rate):
        #Calculate the time necessary for the reference to move into a particular z-window
        #Gather the coordinates of the tracer
        (x_ref, y_ref, z_ref) = self.get_tracer_coordinates(grofile, tracer)

        distance_to_traverse = np.abs(float(z_window) - float(z_ref)) #nm
        t_pulling = distance_to_traverse / pull_coord_rate #ps

        return t_pulling

    def calc_pulling_rate(self, grofile, tracer, z_window, t_pulling):
        (x_ref, y_ref, z_ref) = self.get_tracer_coordinates(grofile, tracer)

        distance_to_traverse = np.abs(float(z_window) - float(z_ref)) #nm
        pull_rate = distance_to_traverse/t_pulling #nm/ps
        return pull_rate

    def get_tracer_coordinates(self, grofile, tracer):
        """ 

        Parameters
        ----------
        grofile : str
            Filename of structure file
        tracer : int
            Molecule number

        Notes
        -----
        Gromacs is 1-indexed, while mdtraj is 0-indexed
        Getting the coordinates of gmx resid 3 means
        getting the coordinates of mdtraj resid 2
            """
        traj = mdtraj.load(grofile)
        tracer_atoms = traj.topology.select('resid {}'.format(tracer-1))
        sub_traj = traj.atom_slice(tracer_atoms)
        com = mdtraj.compute_center_of_mass(sub_traj)

        return [com[0,0], com[0,1], com[0,2]]
    
    def write_ndx(self, grofile, pull_filename, pull_group_names, tracerlist):
        """ Write index file for each pulling group

        Parameters
        ----------
        grofile : str
            Structure file
        pull_filename : str
            name of index file
        pull_group_names : list(str)
            list of group names for index file
        tracerlist: list(int)
            list of tracers (water molecule numbers)
            """

        outfile = open(str(pull_filename[:-4] + '.ndx'), 'w')
        
        tracer_atoms = []
        for i, tracer in enumerate(tracerlist):
            traj = mdtraj.load(grofile)
            tracer_atoms.append(traj.topology.select('resid {}'.format(tracer-1)))

        ##Loop through tracer names and oxygen indices
        for i, groupname in enumerate(pull_group_names):
            outfile.write('[ {} ]\n'.format(groupname))
            outfile.write('{:10.0f}\t{:10.0f}\t{:10.0f}\n'.format(
                float(tracer_atoms[i][0]+1), float(tracer_atoms[i][1]+1), float(tracer_atoms[i][2]+1)))

    def write_tracerlist(self, tracer_list, tracerlog = 'tracers.out'):
        outfile = open(tracerlog, 'w')
        for i, tracer in enumerate(tracer_list):
            outfile.write('{} \n'.format(tracer))
        outfile.close()

    
 
  
    def write_grompp_file(self, directoryname=None, filename=None, 
            grofile=None, mdpfile=None, indexfile=None, topfile=None): 
        gromppfilename = '{}/Grompp_{}.sh'.format(directoryname, filename)
        outfile = open(gromppfilename, 'w')
        outfile.write('gmx grompp -f {} -c {} -p {} -n {} -o {} -maxwarn 2 > grompp_{}.log 2>&1'.format((mdpfile), (grofile),
          topfile, (indexfile), (filename+'.tpr'), filename))

        outfile.close()

    def update_resnames(self,ingro="", outgro="Stage0.gro", outtop="Stage0.top"):
        """ Create new grofile with updated resnames for tracers

        Parameters
        ---------
        ingro : str
            Input gromacs file
        outgro: str
            Filename for new gro file with updated residues
        outtop : str
            FIlename for new top file with updated residues

        """
          
        traj = mdtraj.load(ingro)
        topol = traj.topology
        residues = [b for b in topol.residues]
        atoms = [a for a in topol.atoms]
        # MDTraj renames OW and HW from spc water as O and H, need to put it back
        for atom in atoms:
            if 'HOH' in atom.residue.name or 'SOL' in atom.residue.name:
                atom.name=atom.name[:1]+"W"+atom.name[1:]
        # Rename tracer residues to distinguish position restratining
        for res in residues:
            if 'HOH' in res.name:
                res.name="SOL"
            if res.resSeq in self.tracer_list:
                res.name="TRC"
        traj.save(outgro)
        with open(outtop,'w') as f:
            f.write("""#include "/raid6/homes/ahy3nz/Programs/setup/FF/gromos53a6/ff.itp"  
; Include SPC water topology  
#include "gromos53a6.ff/spc.itp"  
#include "/raid6/homes/ahy3nz/Programs/setup/FF/gromos53a6/trc.itp"

[ system ]
All-atom bilayer system

[ molecules ] 
""")

            counter = 0
            previous_resname = None
            for res in residues:
                # If the new residue name matches old residue, increment tally
                if res.name == previous_resname:
                    counter +=1
                # If not, then we have to print it to the top file
                # But if it's blank then ignore it
                elif not previous_resname:
                    counter = 1
                else: 
                    f.write("{} \t {}\n".format(previous_resname, counter))
                    counter = 1
                previous_resname = res.name
            f.write("{} \t {}\n".format(previous_resname, counter))
