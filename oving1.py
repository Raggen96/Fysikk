import numpy as np
import math as m
import matplotlib.pyplot as plt


c1 = 3.00
c2= 0.80

def v(t):
    return 0.5*c1*t**2-0.25*c2*t**4


t= np.linspace(0,2,1000)

plt.figure(1)
plt.plot(t,v(t))
plt.show()

#Finne areal
v=v(t)
dt = 
