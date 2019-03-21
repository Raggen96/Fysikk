import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
plt.style.use('bmh')    # Nicer looking plots

####################
# fra numfy
####################
# Properties of the rolling object
r = 0.01                 # m      (radius)
#rho = 7850               # kg/m^3 (density)
g = 9.81                 # m/s^2  (gravitational acceleration)


c1 = 2/3            # kuleskall
c2 = 2/5            # kule
c3 = 1              # tynn sirkulær skive

I0 = c*m*r**2            # kg m^2 (moment of inertia)

########################333
def iptrack(filename):
	data=np.loadtxt(filename,skiprows=2)
	return np.polyfit(data[:,1],data[:,2],15), data[0,1]

p, x_0 = iptrack(track)
v_0 = 0
def find_x(x_0):
    x_points = []
    v_points = []
    x_points[0]=x_0
    v_points[0] = 0

    #h = steglengden
    h =
    sluttid-starttid/antall x-punkter
    for i in range (1,N+1):
        x_points[i] = x_points[i-1] + h

    return







