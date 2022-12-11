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
tf = 0.05*tyear # d
dt = 0.08 # d

Temp_day = 7 #C
tday = 1 #day

# Depth, width, length (need to keep dt < a**2/(2*alpha))
D = 10 # m
Nd = 40
W = 50 # m
Nw = 50
L = 4e6 # m
Nl = 100
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

if ( (dt*alpha/ad**2 > 0.5) or (dt*alpha/al**2 > 0.5)):
    dt = 0.5*min(ad,al)**2/alpha
    print('dt adjusted to {}'.format(dt))
    sys.stdout.flush()
for t in np.arange(ti, tf+dt, dt):
    # periodic heating at surface with phase shift (INCORRECT ON SUB-DAY TIMESCALE)
    phase = np.arange(Nl)/Nl
    crust[0] = Tinit + Tvar*np.sin(2*np.pi*(t/tyear+phase)) + Temp_day*np.sin(2*np.pi*(t/tday + phase))
    if (ThisTask == 0 ):
        print('index_low, index_high = ', index_low,index_high)
        sys.stdout.flush()
        for i in range(1, Ntasks):
            print('index_low, index_high = {},{}'.format( index_low+(i*del_index),index_high+(i*del_index)))
            sys.stdout.flush()
            #print('Shape of crust sent = {}'.format(np.shape(crust[:, index_low+(i*del_index) : index_high+(i*del_index)])))
            print('size of data sent in bytes  = {}'.format(sys.getsizeof(crust[:, index_low+(i*del_index) : index_high+(i*del_index)])))
            sys.stdout.flush()
            comm.send(crust[:, index_low+(i*del_index) : index_high+(i*del_index)].copy(), dest = i, tag = i)
            
    else:
        #crust_thistask = np.full((Nd, del_index),0, dtype = 'float')
        #print('Size of crust_thistask in bytes = {}'.format(sys.getsizeof(crust_thistask)))
        #sys.stdout.flush()
        crust_thistask = comm.recv(source = 0, tag = ThisTask)
        print('NON-zero elemenst in crust_thistask is {} in task {} = '.format(np.count_nonzero(crust_thistask),ThisTask))
        sys.stdout.flush()
        print('size of data received in bytes  = {}'.format(sys.getsizeof(crust_thistask)))
        sys.stdout.flush()
            
