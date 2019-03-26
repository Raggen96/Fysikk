import numpy as np
import matplotlib.pyplot as plt


def iptrack(filename):
    # ta bare den første, skal bare ha et utrykk for banen
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:, 1], data[:, 2], 15)


p = iptrack(r'C:\Users\Elise\Documents\6. semester\Fysikk\liten ball\86.txt')

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
    R = (1.0 + dydx ** 2) ** 1.5 / d2ydx2
    return [y, dydx, d2ydx2, alpha, R]


g = 9.81

############   v_n+1   ##################

t_0 = 0
v_0 = 0

t = np.zeros(N + 1)
v = np.zeros(N + 1)

t[0] = t_0
v[0] = v_0

t_old = t_0
v_old = v_0

for n in range(N):
    val = trvalues(p, v_old)
    v_new = v_old + h * (g * np.sin(val[3]) / (1 + c2))

    t[n + 1] = t_old + h
    v[n + 1] = v_new

    t_old = t_old + h
    v_old = v_new


print(r'x_N = %f' % v_old)

###############   x_n+1   ###############

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


def x_av_t(filename):
    x = []
    y = []
    t = []
    v = []
    data = np.loadtxt(filename, skiprows=3)
    for line in data:
        t.append(line[0])
        x.append(line[1])
        y.append(line[2])
        v.append(line[3])
    return [t, x, y, v]


punkter = x_av_t(r'C:\Users\Elise\Fysikk\Fysikk\LB.txt')


def plot_xt():
    plt.figure()
    plt.ylabel(r'$x(t)$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.plot(t2, x, color='#4daf4a')
    plt.plot(punkter[0], punkter[1], color='#377eb8')
    plt.show()



#plot_xt()


v2 = []
for number in v:
    v2.append(number*3.6)

def plot_vt():
    plt.figure()
    plt.ylabel(r'$v(t)$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.plot(t, v2, color='#4d0000')
    plt.plot(punkter[0], punkter[3], color='#007e00')
    plt.show()

print (v)
print(v2)
print(punkter[3])
plot_vt()



________________________________________--

def trvalues(p, x_old):
    tValAll = []
    for i in p:
        y = np.polyval(p[n], x_old)
        dp = np.polyder(p[n])
        dydx = np.polyval(dp, x_old)
        ddp = np.polyder(dp)
        d2ydx2 = np.polyval(ddp, x_old)
        alpha = np.arctan(-dydx)
        R = (1.0+dydx**2)**1.5/d2ydx2
        tValAll.expend(y, dydx, d2ydx2, alpha, R)
    return tValAll