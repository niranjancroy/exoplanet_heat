# exoplanet_heat
This repo is dedicated to the code development for the final project of the Computational Physics course (PHY 5350) at UConn. 

As heat diffusion is a key physical process relevant to almost every field of physics, solving the heat diffusion equation is crucial to effectively model many physical systems. In this work, we build a two dimensional model for the planet Proxima Centauri b and numerically solve the heat equation for the planet's surface.

The "heat_ftcs.py" is the serial version of the solver which performs the 'Forward in Time Central in Space (FTCS)' method of numerically solving the heat equation in 2D and also visualizes the results in 3D by extending the surface to a perpendicular direction. 

The "heat_ftcs_parallel.py" is the parallel version of the solver. It is based on communication among different cells of the simulation domain that are distributed among different cores. As a result, it calculate the updated values of FTCS algorithm at each of the timesteps significantly faster than the serial one when multiple cores are used. It is based on MPI using the mpi4py module and can be run with the following command:

'mpirun -np 4 python heat_ftcs_parallel.py'


Disclaimer: some of the features of this code are still in test phase and are discouraged to be used in scientific study if used as it is.

-Elias Oakes and Niranjan Roy, December 2022
 
