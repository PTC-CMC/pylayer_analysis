import pdb
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.waterdynamics
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt

######################
## Use mdanalysis to study rotational relaxation of water molecules
## Close to the interface
## Note the use of a mol2 to remember accurate resnames (pdb bad)
## and to remember bonds (gro bad)
#####################

def main():
    uni = mda.Universe('extra.mol2', 'npt_80-100ns.xtc', in_memory=True)
    #uni = mda.Universe('extra.mol2', 'last5ns.xtc', in_memory=True)
    midplane = uni.dimensions[2]/2
    top_phosphates = [a for a in uni.atoms if 'P' in a.name[0] 
                        and a.position[2] > midplane]
    bot_phosphates = [a for a in uni.atoms if 'P' in a.name[0] 
                        and a.position[2] < midplane]
    bot_interface = np.mean([a.position[2] for a in bot_phosphates])
    top_interface = np.mean([a.position[2] for a in top_phosphates])

    frame_to_time = 10 # ps
    corr_length = 500 # frames
    selection = '(resname HOH or resname SOL) and prop z <= {0} and prop z >= {1}'.format(
                            top_interface+10, bot_interface-10)
    # Water orientational relaxation
    wor_analysis = mda.analysis.waterdynamics.WaterOrientationalRelaxation(uni, 
                    selection, 0, uni.trajectory.n_frames, corr_length)
    wor_analysis.run()

    #now we print the data ready to plot. The first two columns are WOR_OH vs t plot,
    #the second two columns are WOR_HH vs t graph and the third two columns are WOR_dip vs t graph
    #for WOR_OH, WOR_HH, WOR_dip in wor_analysis.timeseries:
        #print("{time} {WOR_OH} {time} {WOR_HH} {time} {WOR_dip}".format(time=time, WOR_OH=WOR_OH, WOR_HH=WOR_HH,WOR_dip=WOR_dip))


    frames = np.arange(0, corr_length, dtype=int) 
    times = frames * frame_to_time
    oh_timeseries =[column[0] for column in wor_analysis.timeseries]
    hh_timeseries =[column[1] for column in wor_analysis.timeseries]
    dip_timeseries =[column[2] for column in wor_analysis.timeseries]
    np.savetxt('oh_corr.dat',np.column_stack((frames, times, oh_timeseries)), 
            header="frames, times, oh_corr")

    np.savetxt('hh_corr.dat',np.column_stack((frames, times, hh_timeseries)), 
            header="frames, times, hh_corr")

    np.savetxt('dip_corr.dat',np.column_stack((frames, times, dip_timeseries)), 
            header="frames, times, dip_corr")



    plt.figure(1,figsize=(18, 6))

    #WOR OH
    plt.subplot(131)
    plt.xlabel('time')
    plt.ylabel('WOR')
    plt.title('WOR OH')
    plt.plot(times,[column[0] for column in wor_analysis.timeseries])

    #WOR HH
    plt.subplot(132)
    plt.xlabel('time')
    plt.ylabel('WOR')
    plt.title('WOR HH')
    plt.plot(times,[column[1] for column in wor_analysis.timeseries])

    #WOR dip
    plt.subplot(133)
    plt.xlabel('time')
    plt.ylabel('WOR')
    plt.title('WOR dip')
    plt.plot(times,[column[2] for column in wor_analysis.timeseries])

    plt.savefig('wor.png')

if __name__ == "__main__":
    main()
