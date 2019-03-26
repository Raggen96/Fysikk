#from sympy import *
import numpy as np
import matplotlib.pyplot as plt


def iptrack(filename):
    #ta bare den første, skal bare ha et utrykk for banen
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:, 1], data[:, 2], 15)


pLB = iptrack(r'C:\Users\Elise\Documents\6. semester\Fysikk\LBU.txt')
pPP = iptrack(r'C:\Users\Elise\Documents\6. semester\Fysikk\PPU.txt')
pR = iptrack(r'C:\Users\Elise\Documents\6. semester\Fysikk\RU.txt')

pAll = [pLB, pPP, pR]
g = 9.81


# Set common figure parameters
newparams = {'figure.figsize': (16, 6), 'axes.grid': True,
             'lines.linewidth': 1.5, 'lines.markersize': 10,
             'font.size': 14}
plt.rcParams.update(newparams)

N = 500  # number of steps
h = 0.001  # step size


def trvalues(p, x_old):
    y = np.polyval(p, x_old)
    dp = np.polyder(p)
    dydx = np.polyval(dp, x_old)
    ddp = np.polyder(dp)
    d2ydx2 = np.polyval(ddp, x_old)
    alpha = np.arctan(-dydx)
    R = (1.0+dydx**2)**1.5/d2ydx2
    return [y, dydx, d2ydx2, alpha, R]



t_0 = 0
v_0 = 1.135
t = np.zeros(N + 1)
v = np.zeros(N + 1)
t[0] = t_0
v[0] = v_0
t_old = t_0
v_old = v_0
for n in range(N):
    val = trvalues(p, x_old)
    v_new = v_old + h * (g*np.sin(val[3])/(1+c2))
    t[n + 1] = t_old + h
    v[n + 1] = v_new

    t_old = t_old + h
    v_old = v_new
#print(r'x_N = %f' % v_old)


t2_0 = 0
x_0 = -0.427
t2 = np.zeros(N + 1)
x = np.zeros(N + 1)
t2[0] = t2_0
x[0] = x_0
t2_old = t2_0
x_old = x_0

for n in range(N):
    val = trvalues(p, x_old)
    x_new = x_old + h * (v[n])
    t2[n + 1] = t2_old + h
    x[n + 1] = x_new
    t2_old = t2_old + h
    x_old = x_new


def x_av_t(filename):   # gir eksperementielle verdier for x v t
    xpos = []
    ypos = []
    tpos = []
    data = np.loadtxt(filename, skiprows=2)
    for line in data:
        tpos.append(line[0])
        xpos.append(line[1])
        ypos.append(line[2])
    return [xpos, ypos, tpos]


punkterLB = x_av_t(r'C:\Users\Elise\Documents\6. semester\Fysikk\LB.txt')
punkterPP = x_av_t(r'C:\Users\Elise\Documents\6. semester\Fysikk\PP.txt')
punkterR = x_av_t(r'C:\Users\Elise\Documents\6. semester\Fysikk\R.txt')

punkterAll = [punkterLB, punkterPP, punkterR]


def plot_xt():
    plt.figure()
    plt.ylabel(r'$x(t)$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.plot(t2, x, color = '#4daf4a')
    plt.plot(punkter[2], punkter[0], color = '#377eb8')
    plt.show()


def plot_vt():
    #euler_vt()
    plt.figure()
    plt.ylabel(r'$v(t)$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.plot(t, v, color='#4daf4a')
    plt.plot(punkter[2], vt, color='#377eb8')
    plt.show()


cPP = 2 / 3
cLB = 2 / 5
cR = 1
cAll = [cLB, cPP, cR]
plot_xt()


for n in range (3):
    punkter = punkterAll[n]
    c = cAll[n]
    p = pAll[n]
    plot_vt()



