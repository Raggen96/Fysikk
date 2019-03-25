import numpy as np
import matplotlib.pyplot as plt

# Set common figure parameters
newparams = {'figure.figsize': (16, 6), 'axes.grid': True,
             'lines.linewidth': 1.5, 'lines.markersize': 10,
             'font.size': 14}
plt.rcParams.update(newparams)

def iptrack(filename):
    #ta bare den første, skal bare ha et utrykk for banen
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:, 1], data[:, 2], 15)
p = iptrack(r'C:\Users\hkgol\PycharmProjects\Fysikk\litenball94.txt')

def trvalues(p,x_old):
    y = np.polyval(p, x_old)
    dp = np.polyder(p)
    dydx = np.polyval(dp, x_old)
    ddp = np.polyder(dp)
    d2ydx2 = np.polyval(ddp, x_old)
    alpha = np.arctan(-dydx)
    R =(1.0+dydx**2)**1.5/d2ydx2
    return [y, dydx, d2ydx2, alpha, R]


N = 500  # number of steps
h = 0.001  # step size

# initial values
t = np.zeros(N + 1)
x = np.zeros(N + 1)
v = np.zeros(N + 1)

t[0] = t_0 = t_old = 0
x[0] = x_0 = x_old = (- 0.427)
v[0] = v_0 = v_old = 0

c1 = 2/3
c2 = 2/5
c3 = 1

g = 9.81





#teoretisk fart vha Euler

for n in range(N):

    x_new = x[0] + v[n]*h

    y, dydx, d2ydx2, alpha, R = trvalues(p, x_new)

    v_new = v[n] + (h * (g*np.sin(R)/(1+c2)))  # Euler's method

    t[n + 1] = t[n] + h
    x[n + 1] = x_new
    v[n + 1] = v_new

plt.figure()
plt.plot(t, v)  # plotting the velocity vs. time: v(t)
plt.xlabel(r'$t$')
plt.ylabel(r'$v(t)$')
plt.grid()
plt.show()

xer = np.zeros(N+1)
yer = np.zeros(N+1)
ver = np.zeros(N+1)
ter = np.zeros(N + 1)

#eksperimentell fart
for i in range(N):
    x1 = xer[0] + (i * h)
    t
    xer.append(x1)

    y1, dydx1, d2ydx21, alpha1, R1 = trvalues(p, x1)
    yer.append(y1)
    ver.append(ver)

    xer[n + 1] = x1

plt.figure()
plt.plot(t, v)  # plotting the velocity vs. time: v(t)
plt.xlabel(r'$t$')
plt.ylabel(r'$v(t)$')
plt.grid()
plt.show()












