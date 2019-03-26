#from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt


def iptrack(filename):
    #ta bare den første, skal bare ha et utrykk for banen
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:, 1], data[:, 2], 15)


pLB = iptrack(r'./LBU.txt')
pPP = iptrack(r'./PPU.txt')
pR = iptrack(r'./RU.txt')

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
    #print('y:', y, 'dydx:', dydx, 'd2ydx2', d2ydx2, 'alpha', alpha, 'R', R)
    return [y, dydx, d2ydx2, alpha, R]


t_0 = 0
t = np.zeros(N + 1)
v = np.zeros(N + 1)
def euler_vt():
    t[0] = t_0
    v[0] = v_0
    t_old = t_0
    v_old = v_0
    for n in range(N):
        x_o = x[n]
        val = trvalues(p, x_o)
        v_new = v_old + h * (g*np.sin(val[3])/(1+c))
        t[n + 1] = t_old + h
        v[n + 1] = v_new

        t_old = t_old + h
        v_old = v_new
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
        #val = trvalues(p, x_old)
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


punkterLB = x_av_t(r'./LB.txt')
punkterPP = x_av_t(r'./PP.txt')
punkterR = x_av_t(r'./R.txt')

punkterAll = [punkterLB, punkterPP, punkterR]


def plot_xt():
    plt.figure(0)
    plt.plot(t2, x, label=labelNum)
    plt.plot(p0, p2, label=labelEks)

def plot_vt():
    plt.figure(1)
    #print ("t-verdier:", t)
    #print("v-verdier: ", v)
    plt.plot(t, v, label=labelNum)
    plt.plot(p0, p1, label=labelEks)


cPP = 2 / 3
cLB = 2 / 5
cR = 1
cAll = [cLB, cPP, cR]
labelAllEks = ["Kule eksperimentiell", "Kuleskall eksperimentiell", "Ring eksperimentiell"]
labelAllNum = ["Kule teoretisk", "Kuleskall teoretisk", "Ring teoretisk"]

v_0All = [1.1348810766934287, 1.1470505158475683, 1.0590491123276344]

t_values = [[0], [0], [0]]

test = []
v_values = []
t2_values = [[0], [0], [0]]
x_values = [[0], [0], [0]]

# Prøver å legge t, v og x verdiene inn i en liste sånn at jeg kan plotte alle. Når jeg printer v inni for løkken
# får jeg forskjellige resultater hver for hver loop, men nnår jeg prøver å legge inn i v_values listen blir alle
# verdiene like, skjønner ikke hvorfor
for n in range(3):
    v_0 = v_0All[n]
    labelEks = labelAllEks[n]
    labelNum = labelAllNum[n]
    p0 = punkterAll[n][0]
    p1 = punkterAll[n][1]
    p2 = punkterAll[n][2]
    p3 = punkterAll[n][3]
    c = cAll[n]
    p = pAll[n]
    euler_xt()
    plot_xt()
    euler_vt()
    plot_vt()
    # print ('Resultat av v for', n, v)
    v_values.append(v)
    print( 'resultat for v: ', v, 'resultaterrr')

print (v_values)

#print("v-verdier", v_values)
# plt.figure()

plt.figure(0)
plt.ylabel(r'$x(t)$')
plt.xlabel(r'$t$')
plt.grid()
plt.legend()

plt.figure(1)
plt.ylabel(r'$v(t)$')
plt.xlabel(r'$t$')
plt.grid()
plt.legend()

# plt.ylabel(r'$v(t)$')
# plt.xlabel(r'$t$')
# plt.grid()

#plt.plot(t, v, color='#4daf4a', label=labelNum)
#plt.plot(p0, p1, color='#377eb8', label=labelEks)
tv0 = t_values[0]
vv0 = v_values[0]
tv1 = t_values[1]
vv1 = v_values[1]
tv2 = t_values[2]
vv2 = v_values[2]
#plt.plot(t_values[0], v_values[0])
#plt.plot(t_values[1], v_values[1], label='test2')
#plt.plot(t_values[2], v_values[2], label='test3')
# plt.legend()
plt.show()





