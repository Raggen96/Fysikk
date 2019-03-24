

import numpy as np
import matplotlib.pyplot as plt


def iptrack(filename):
	#ta bare den første, skal bare ha et utrykk for banen
	data = np.loadtxt(filename, skiprows=2)
	return np.polyfit(data[:, 1], data[:, 2], 15)

p = iptrack(r'C:\Users\Elise\Documents\6. semester\Fysikk\New\liten ball\liten ball\86.txt')




# Set common figure parameters
newparams = {'figure.figsize': (16, 6), 'axes.grid': True,
             'lines.linewidth': 1.5, 'lines.markersize': 10,
             'font.size': 14}
plt.rcParams.update(newparams)

N = 500  # number of steps
h = 0.001  # step size

# initial values
t_0 = 0
x_0 = -0.427

t = np.zeros(N + 1)
x = np.zeros(N + 1)

t[0] = t_0
x[0] = x_0

t_old = t_0
x_old = x_0

def trvalues(p,x_old):
    y = np.polyval(p, x_old)
    dp = np.polyder(p)
    dydx = np.polyval(dp, x_old)
    ddp = np.polyder(dp)
    d2ydx2 = np.polyval(ddp, x_old)
    alpha = np.arctan(-dydx)
    R =(1.0+dydx**2)**1.5/d2ydx2
    return [y, dydx, d2ydx2, alpha, R]

c1 = 2/3
c2= 2/5
c3 = 1
g = 9.81

###############   x_n+1   ###############

t2_0 = 0
x2_0 = -0.427

t2 = np.zeros(N + 1)
x2 = np.zeros(N + 1)

t2[0] = t2_0
x2[0] = x2_0

t2_old = t2_0
x2_old = x2_0


for n in range(N):
    val = trvalues(p, x2_old)
    x2_new = x2_old + h * (val[1])  # Euler's method

    t2[n + 1] = t2_old + h
    x2[n + 1] = x2_new

    t2_old = t2_old + h
    x2_old = x2_new




##############################

for n in range(N):
    val = trvalues(p, x_old)
    print (val)
    x_new = x_old + h * (g*np.sin(val[3])/(1+c2))  # Euler's method

    t[n + 1] = t_old + h
    x[n + 1] = x_new

    t_old = t_old + h
    x_old = x_new

print(r'x_N = %f' % x_old)

# Plot x(t)
#plt.figure()
#plt.plot(t, x)
#plt.ylabel(r'$x(t)$')
#plt.xlabel(r'$t$')
#plt.grid()
#plt.show()

test = [0,1]

def x_av_t(filename):
    x= []
    y = []
    t = []
    data = np.loadtxt(filename, skiprows=2)
    for line in data:
        x.append(line[1])
        y.append(line[2])
        t.append(line[0])
    return [x, y, t]
punkter = x_av_t(r'C:\Users\Elise\Documents\6. semester\Fysikk\New\liten ball\liten ball\86.txt')


for i in test:
    plt.plot(t2,x2, color = '#4daf4a')
    plt.plot(punkter[2], punkter[0], color = '#377eb8')
    plt.show()


