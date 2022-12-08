#import necessary libraries and custom plotting functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
from plotting_functions import * 
plt.rc('text', usetex=True)


def get_grid(mx, my, mz, Lx,Ly,Lz):
    #print('mx,my,mz = ',mx,my,mz)
    ix, iy, iz = Lx*np.linspace(0,1,mx), Ly*np.linspace(0,1,my), Lz*np.linspace(0,1,mz)
    x, y, z = np.meshgrid(ix,iy,iz, indexing='ij')
    #print('ix', ix), print('iy', iy), print('iz', iz)
    return x,y,z

def plot_grid(x,y,z,T, t, filename):
    def plot_boundary_only(x,y,z,T):
        #mx, my, mz = x.shape
        x[1:-1, 1:-1, 1:-1],y[1:-1, 1:-1, 1:-1],z[1:-1, 1:-1, 1:-1],T[1:-1, 1:-1, 1:-1] = np.nan, np.nan, np.nan, np.nan \
                #This removes interior because we cannot see it anyway? reduces time to plot
        return x,y,z,T
    
    x,y,z,T = plot_boundary_only(x,y,z,T)   
    fig = plt.figure(figsize=(15,15), facecolor = 'w')
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x,y,z, c=T.reshape(-1), s=150, vmin = -2.5, vmax = 23, cmap=plt.inferno())
    cbar = fig.colorbar(img, orientation='horizontal', fraction=0.047, pad=0.15, aspect=15)
    plt.tick_params(axis='both', which='major', labelsize=18)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Temperature ($^oC$)', size = 22)
    
    ax.text(0,0,-15, 't = %i days'%t, fontsize=20)
    ax.set_xticklabels(['%i'%(l*W/Nw) for l in ax.get_xticks()])
    ax.set_yticklabels(['%i'%(l*L/Nl/1e3) for l in ax.get_yticks()])
    ax.set_zticklabels(['%i'%(l*D/Nd) for l in ax.get_zticks()])
    ax.set_ylabel('Y (km)', labelpad=20, fontsize=20)
    ax.set_xlabel('X (m)', labelpad=20, rotation=0, fontsize=20)
    ax.set_zlabel('Depth from surface (m)', rotation=0, fontsize=20)
    ax.invert_zaxis()
    ax.view_init(15, -20)
    #plt.tight_layout()
    #plt.savefig('tempevolution_plots/temp%s.png'%count)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
def init_T(x,y,z, T_3d):
    #print('size of y = ', np.shape(y))
    T = np.zeros_like(x)
    #print('Size of T = ',np.shape(T))
    T = T_3d
    return T

def show_plot(crust, t):
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    im = ax.imshow(crust, vmin=0, vmax=20, cmap = 'inferno', aspect = 1 * (Nw/Nd))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('$\mathrm{Temperature\ [K]}$', rotation=270, labelpad=15)
    ax.set_title('$\mathrm{t = %i\ days}$'%t)
    ax.set_xlabel('$\mathrm{Width}$')
    ax.set_ylabel('$\mathrm{Depth}$')


# Constants (currently set for Earth vals)

Tinit = 10 # C
Tvar = 12 # C
Tbottom = 11 # C

# Diffusion constant (https://www.nature.com/articles/nature07818)
#alpha = 0.0864 # m^2 / day
alpha = 0.389 #m^2 / day (https://terra-docs.s3.us-east-2.amazonaws.com/IJHSR/Articles/volume4-issue3/2022_43_p17_Murgia.pdf)

# times 
tyear = 11.2 # days in a year of proxima b
ti = 0 # d
tf = 20*tyear # d
dt = 0.08 # d

Tday = 7
tday = 1

# Depth, width, length (need to keep dt < a**2/(2*alpha))
D = 10 # m
Nd = 40
W = 50 # m
Nw = 50
L = 4e6 # m
Nl = 100
ad = D/Nd
al = L/Nl

# Initialize "crust": 2D array of T = Tinit except at bottom
crust = np.full((Nd,Nl), Tinit, dtype='float')
crust[Nd-1] = Tbottom

count = 0

if ( (dt*alpha/ad**2 > 0.5) or (dt*alpha/al**2 > 0.5)):
        dt = 0.5*min(ad,al)**2/alpha
        print('dt adjusted to {}'.format(dt))

for t in np.arange(ti, tf+dt, dt):
    
    # periodic heating at surface with phase shift (INCORRECT ON SUB-DAY TIMESCALE)
    phase = np.arange(Nl)/Nl
    crust[0] = Tinit + Tvar*np.sin(2*np.pi*(t/tyear+phase)) + Tday*np.sin(2*np.pi*(t/tday + phase))
    
    # depth (skipping top and bottom to maintain boundaries)
    for i in range(1, Nd-1):
        # width (periodic boundary connecting 0 to Nw)
        for j in range(0, Nl-1):
            # evolve crust with FTCS approach! (in 2D)
            
            if j == Nl-1: # to avoid overflow at crust[i,j+1]
                crust[i,j] = crust[i,j] + dt*alpha*(
                    (crust[i+1,j] + crust[i-1,j] - 2*crust[i,j])/ad**2 + 
                    (crust[i,0] + crust[i,j-1] - 2*crust[i,j])/al**2)
            else:
                crust[i,j] = crust[i,j] + dt*alpha*(
                    (crust[i+1,j] + crust[i-1,j] - 2*crust[i,j])/ad**2 + 
                    (crust[i,j+1] + crust[i,j-1] - 2*crust[i,j])/al**2)
    
    # Show plot

    if t%(tyear//5) == 0:
        crust_3d = np.zeros((Nw,Nl,Nd))

        for i in range(Nw):
            crust_3d[i,:,:] = crust.T
        
        #show_plot(crust, t)
        #plt.show()
        nx, ny, nz = Nw, Nl, Nd
        Lx, Ly, Lz = nx-1, ny-1, nz-1
        x,y,z = get_grid(nx, ny, nz, Lx,Ly,Lz)  # generate a grid with mesh size Δx = Δy = Δz = 1
        T = init_T(x,y,z, crust_3d)
        filename = 'tempevolution_plots/proxima_b_test/temp'+str(count).zfill(3)+'.png'
        plot_grid(x,y,z,T, t, filename)
        #print('File {} saved \n'.format(filename))
        count += 1

#creating the animation
#print('Creating animation...')
#os.system('ffmpeg -i tempevolution_plots/temp%03d.png  -c:v libx264 -r 8 -pix_fmt yuv420p -vf \
#                    "scale=trunc(iw/2)*2:trunc(ih/2)*2" tempevolution_plots/surface_temp_final.mp4')
#print('animation created and saved')
