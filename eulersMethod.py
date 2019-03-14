import numpy as np
from matplotlib import pyplot as plt

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







