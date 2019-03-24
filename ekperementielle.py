import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
plt.style.use('bmh')    # Nicer looking plots


# tidsposisjonsgraf

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


x_av_t(r'C:\Users\Elise\Documents\6. semester\Fysikk\New\liten ball\liten ball\86.txt')

