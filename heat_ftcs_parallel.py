import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
plt.rc('text', usetex=True)
from mpi4py import MPI
import sys 



comm  = MPI.COMM_WORLD
Ntasks = comm.Get_size()
ThisTask = comm.Get_rank()


# Constants (currently set for Earth vals)

Tinit = 10 # C
Tvar = 12 # C
Tbottom = 11 # C

# Diffusion constant (https://www.nature.com/articles/nature07818)
alpha = 0.0864 # m^2 / day
#alpha = 0.389 #m^2 / day (https://terra-docs.s3.us-east-2.amazonaws.com/IJHSR/Articles/volume4-issue3/2022_43_p17_Murgia.pdf)

# times
tyear = 11.2 # days in a year of proxima b
ti = 0 # d
tf = 0.01*tyear # d
dt = 0.08 # d

Temp_day = 7 #C
tday = 1 #day

# Depth, width, length (need to keep dt < a**2/(2*alpha))
D = 10 # m
Nd = 40
W = 50 # m
Nw = 50
L = 4e6 # m
Nl = 1000
ad = D/Nd
al = L/Nl




#We will be deviding the 2D array columwise among the tasks

index_high = int(((ThisTask + 1) * Nl) / Ntasks) #setting upper bound of column no for each task
index_low = int((ThisTask * Nl) / Ntasks)  #setting lower bound of column no for each task
del_index = int(Nl / Ntasks) #For now please provide Ntasks such that del_index is automatically int
print('del_index = {}'.format(del_index))
sys.stdout.flush()

crust = np.full((Nd,Nl), Tinit, dtype='float')
#crust = np.ones((Nd,Nl), dtype = 'float') * Tinit
crust[Nd-1] = Tbottom

#initializing the cells
cell_left = np.zeros(1)
cell_right = np.zeros(1)
#cell_upper = np.zeros(1)
#cell_lower = np.zeros(1)


if ( (dt*alpha/ad**2 > 0.5) or (dt*alpha/al**2 > 0.5)):
    dt = 0.5*min(ad,al)**2/alpha
    print('dt adjusted to {}'.format(dt))
    sys.stdout.flush()
for t in np.arange(ti, tf+dt, dt):
    # periodic heating at surface with phase shift (INCORRECT ON SUB-DAY TIMESCALE)
    phase = np.arange(Nl)/Nl
    crust[0] = Tinit + Tvar*np.sin(2*np.pi*(t/tyear+phase)) + Temp_day*np.sin(2*np.pi*(t/tday + phase))

    crust_thistask = crust[:,index_low:index_high]
    nrow, ncol = np.shape(crust_thistask)
    print('Shape of crust_thistask is= {}, in task = {}'.format(np.shape(crust_thistask), ThisTask))
    sys.stdout.flush()


    if ThisTask == 0 : #rank == 0
        for i in range(1,nrow-1):
            for j in range(0,ncol-1):
                if ((j-1) < index_low):
                    #cell_left = comm.irecv(source = Ntasks-1, tag = Ntasks-1)
                    #print('Left cell received at task {} of {} from task {}'.format(ThisTask, Ntasks, Ntasks-1))
                    #sys.stdout.flush()
                    #cell_right = crust_thistask[i,j+1]
                    cell_left = np.zeros(1)
                    comm.isend(crust_thistask[i,j], dest = Ntasks-1, tag = 0)
                    print('Right cell sent from task {} of {} to task {}'.format(ThisTask, Ntasks, Ntasks-1))
                    sys.stdout.flush()
                    print('Waiting to receive left cell at task {}'.format(ThisTask))
                    cell_left_req = comm.irecv(cell_left, source = Ntasks-1, tag = Ntasks-1)
                    #cell_left_req.wait()
                    print('Left cell received at task {} of {} from task {}'.format(ThisTask, Ntasks, Ntasks-1))
                    sys.stdout.flush()
                    cell_left = cell_left[0]
                    cell_right = crust_thistask[i,j+1]
                    cell_upper = crust_thistask[i+1,j]
                    cell_lower = crust_thistask[i-1,j]
                elif ((j+1) > index_high):
                    cell_left = crust_thistask[i,j-1]
                    print('Waiting to receive left cell at task {}'.format(ThisTask))
                    cell_right = np.zeros(1)
                    cell_right_req = comm.irecv(cell_right, source = 1, tag = 1)
                    #cell_right_req.wait()
                    print('Right cell received at task {} of {} from task {}'.format(ThisTask, Ntasks, 1))
                    sys.stdout.flush()
                    cell_right = cell_right[0]
                    comm.isend(crust_thistask[i,j], dest = 1, tag = 0)
                    print('Left cell sent from task {} of {} to task {}'.format(ThisTask, Ntasks, 1))
                    sys.stdout.flush()
                    cell_upper = crust_thistask[i+1,j]
                    cell_lower = crust_thistask[i-1,j]
                else:
                    cell_upper = crust_thistask[i+1,j]
                    cell_lower = crust_thistask[i-1,j]
                    cell_right = crust_thistask[i,j+1]
                    cell_left = crust_thistask[i,j-1]
                    print('This is not a edge cell at task {}'.format(ThisTask))
                    sys.stdout.flush()
                crust_thistask[i,j] = crust_thistask[i,j] + dt * alpha * \
                                      ( (cell_upper + cell_lower - 2 * crust_thistask[i,j] ) / ad**2 + \
                                      (cell_right + cell_left - 2 * crust_thistask[i,j]) / al**2 )




    elif ThisTask != 0 :
        for i in range(1,nrow-1):
            for j in range(0,ncol-1):
                if ((j-1) < index_low):
                    comm.isend(crust_thistask[i,j], dest=ThisTask-1, tag = ThisTask)
                    print('Right cell sent from task {} of {} to task {}'.format(ThisTask, Ntasks, ThisTask-1))
                    sys.stdout.flush()
                    cell_left = np.zeros(1)
                    print('Waiting to receive left cell at task {}'.format(ThisTask))
                    cell_left_req = comm.irecv(cell_left, source = ThisTask-1, tag = ThisTask-1)
                    #cell_left_req.wait()
                    print('Left cell received at task {} of {} from task {}'.format(ThisTask, Ntasks, ThisTask-1))
                    sys.stdout.flush()
                    cell_left = cell_left[0]
                    cell_right = crust_thistask[i,j+1]
                    cell_upper = crust_thistask[i+1,j]
                    cell_lower = crust_thistask[i-1,j] 
                elif ((j+1) > index_high): #if left cell needs to come from another task
                    if (ThisTask < Ntasks-1):
                        comm.isend(crust_thistask[i,j], dest = ThisTask+1, tag=ThisTask) #This is right edge, send to next task as left cell
                        print('Left cell sent from task {} of {} to task {}'.format(ThisTask, Ntasks, ThisTask+1))
                        sys.stdout.flush()
                        cell_left = crust_thistask[i,j-1]
                        print('Waiting to receive left cell at task {}'.format(ThisTask))
                        cell_right = np.zeros(1)
                        cell_right_req = comm.irecv(cell_right, source = ThisTask+1, tag = ThisTask+1)
                        print('Right cell received at task {} of {} from task {}'.format(ThisTask-1, Ntasks, ThisTask+1))
                        sys.stdout.flush()
                        cell_right = cell_right[0]
                        cell_upper = crust_thistask[i+1,j]
                        cell_lower = crust_thistask[i-1,j]
                    else:
                        comm.isend(crust_thistask[i,j], dest = 0, tag=ThisTask)
                        cell_left = crust_thistask[i,j-1]
                        print('Waiting to receive left cell at task {}'.format(ThisTask))
                        cell_right = np.zeros(1)
                        cell_right_req = comm.irecv(cell_right, source = 0, tag = 0)
                        #cell_right_req.wait()
                        print('Right cell received at task {} of {} from task {}'.format(ThisTask-1, Ntasks, ThisTask+1))
                        sys.stdout.flush()
                        cell_right = cell_right[0]
                        cell_upper = crust_thistask[i+1,j]
                        cell_lower = crust_thistask[i-1,j]
    #            elif ((j-1) < index_low):
    #                comm.isend(crust_thistask[i,j], dest=ThisTask-1, tag = ThisTask)
    #                cell_left = comm.irecv(source = ThisTask-1, tag = ThisTask-1)
    #                cell_right = crust_thistask[i,j+1]
                else:
                    cell_upper = crust_thistask[i+1,j]
                    cell_lower = crust_thistask[i-1,j]
                    cell_right = crust_thistask[i,j+1]
                    cell_left = crust_thistask[i,j-1]
                    print('This is not a edge cell at task {}'.format(ThisTask)) 
                    sys.stdout.flush()
                crust_thistask[i,j] = crust_thistask[i,j] + dt * alpha * \
                                      ( (cell_upper + cell_lower - 2 * crust_thistask[i,j] ) / ad**2 + \
                                      (cell_right + cell_left - 2 * crust_thistask[i,j]) / al**2 )

#    else: #rank == 0
#        for i in range(1,nrow-1):
#            for j in range(0,ncol-1):
#                if ((j-1) < index_low):
#                    cell_left = comm.irecv(source = Ntasks-1, tag = Ntasks-1)  
#                    print('Left cell received at task {} of {} from task {}'.format(ThisTask, Ntasks, Ntasks-1))
#                    sys.stdout.flush()
#                    cell_right = crust_thistask[i,j+1]
#                    comm.isend(crust_thistask[i,j], dest = Ntasks-1, tag = 0)
#                    print('Right cell sent from task {} of {} to task {}'.format(ThisTask, Ntasks, Ntasks-1))
#                    sys.stdout.flush()
#                    cell_upper = crust_thistask[i+1,j]
#                    cell_lower = crust_thistask[i-1,j]
#                elif ((j+1) > index_high):
#                    cell_left = crust_thistask[i,j-1]
#                    cell_right = comm.irecv(source = 1, tag = 1)
#                    print('Right cell received at task {} of {} from task {}'.format(ThisTask, Ntasks, 1))
#                    sys.stdout.flush()
#                    comm.isend(crust_thistask[i,j], dest = 1, tag = 0) 
#                    print('Left cell sent from task {} of {} to task {}'.format(ThisTask, Ntasks, 1))
#                    sys.stdout.flush()
#                    cell_upper = crust_thistask[i+1,j]
#                    cell_lower = crust_thistask[i-1,j]
#
#                crust_thistask[i,j] = crust_thistask[i,j] + dt * alpha * \
#                                      ( (cell_upper + cell_lower - 2 * crust_thistask[i,j] ) / ad**2 + \
#                                      (cell_right + cell_left - 2 * crust_thistask[i,j]) / al**2 )        

#
#    if (ThisTask == 0 ):
#        print('index_low, index_high = ', index_low,index_high)
#        sys.stdout.flush()
#        for row in 
#
#        for i in range(1, Ntasks):
#            print('index_low, index_high = {},{}'.format( index_low+(i*del_index),index_high+(i*del_index)))
#            sys.stdout.flush()
#            #print('Shape of crust sent = {}'.format(np.shape(crust[:, index_low+(i*del_index) : index_high+(i*del_index)])))
#            print('size of data sent in bytes  = {}'.format(sys.getsizeof(crust[:, index_low+(i*del_index) : index_high+(i*del_index)])))
#            sys.stdout.flush()
#            comm.isend(crust[:, index_low+(i*del_index) : index_high+(i*del_index)].copy(), dest = i, tag = i)
#            
#    else:
#        #crust_thistask = np.full((Nd, del_index),0, dtype = 'float')
#        #print('Size of crust_thistask in bytes = {}'.format(sys.getsizeof(crust_thistask)))
#        #sys.stdout.flush()
#        crust_thistask = comm.irecv(source = 0, tag = ThisTask)
#        print('NON-zero elemenst in crust_thistask is {} in task {} = '.format(np.count_nonzero(crust_thistask),ThisTask))
#        sys.stdout.flush()
#        print('size of data received in bytes  = {}'.format(sys.getsizeof(crust_thistask)))
#        sys.stdout.flush()
#            
