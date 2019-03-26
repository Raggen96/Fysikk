#from sympy import *
import numpy as np
import matplotlib.pyplot as plt


def iptrack(filename):
    #ta bare den første, skal bare ha et utrykk for banen
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:, 1], data[:, 2], 15)


pLB = iptrack(r'C:\Users\hkgol\PycharmProjects\Fysikk\LBU.txt')
pPP = iptrack(r'C:\Users\hkgol\PycharmProjects\Fysikk\PPU.txt')
pR = iptrack(r'C:\Users\hkgol\PycharmProjects\Fysikk\RU.txt')

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
def euler_vt():
    t[0] = t_0
    v[0] = v_0
    t_old = t_0
    v_old = v_0
    for n in range(N):
        val = trvalues(p, v_old)
        v_new = v[n] + h * (g*np.sin(val[3])/(1+c))
        t[n + 1] = t[n] + h
        v[n + 1] = v_new

        v[n+1] = v_new
    return [t, v]


t2_0 = 0
x_0 = -0.427
t2 = np.zeros(N + 1)
x = np.zeros(N + 1)
def euler_xt():
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
    return [t2, x]


def x_av_t(filename):   # gir eksperementielle verdier for x v t
    tpos = []
    vpos = []
    xpos = []
    ypos = []

    data = np.loadtxt(filename, skiprows=2)
    for line in data:
        tpos.append(line[0])
        vpos.append(line[1])
        xpos.append(line[2])
        ypos.append(line[3])
    return [tpos, vpos, xpos, ypos]


punkterLB = x_av_t(r'C:\Users\hkgol\PycharmProjects\Fysikk\LB.txt')
punkterPP = x_av_t(r'C:\Users\hkgol\PycharmProjects\Fysikk\PP.txt')
punkterR = x_av_t(r'C:\Users\hkgol\PycharmProjects\Fysikk\R.txt')

punkterAll = [punkterLB, punkterPP, punkterR]


def plot_xt():
    plt.figure()
    plt.ylabel(r'$x(t)$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.plot(t2, x, color = '#4daf4a', label=labelNum)
    plt.plot(p0, p2, color = '#377eb8', label=labelEks)
    plt.legend()
    plt.show()

def plot_vt():
    plt.figure()
    plt.ylabel(r'$v(t)$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.plot(t, v, color='#4daf4a', label=labelNum)
    plt.plot(p0, p1, color='#377eb8', label=labelEks)
    plt.legend()
    plt.show()


cPP = 2 / 3
cLB = 2 / 5
cR = 1
cAll = [cLB, cPP, cR]
labelAllEks = ["Kule eksperimentiell", "Kuleskall eksperimentiell", "Ring eksperimentiell"]
labelAllNum = ["Kule teoretisk", "Kuleskall teoretisk", "Ring teoretisk"]
for n in range(3):
    labelEks = labelAllEks[n]
    labelNum = labelAllNum[n]
    p0 = punkterAll[n][0]
    p1 = punkterAll[n][1]
    p2 = punkterAll[n][2]
    p3 = punkterAll[n][3]
    c = cAll[n]
    p = pAll[n]
    euler_vt()
    plot_vt()
    euler_xt()
    #plot_xt()






