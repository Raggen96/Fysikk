import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
plt.style.use('bmh')    # Nicer looking plots


####### variabler

m = 1

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

I01 = c1*m*r**2            # kg m^2 (moment of inertia)
I02 = c2*m*r**2
I03 = c3*m*r**2

########################


#N = len(yi)               #   (# of mounts)
#xi = np.linspace(0, L, N) # m (x-positions)
#plt.plot(xi, yi)
#plt.show()


#get_y = interp.CubicSpline(xi, yi, bc_type="natural")

# KALKULERER theta
# trvalues - track values
#
# SYNTAX
# [y,dydx,d2ydx2,alpha,R]=trvalues(p,x)
#
# INPUT
# p: the n+1 coefficients of a polynomial of degree n, given in descending
# order. (For instance the output from p=iptrack(filename).)
# x: ordinate value at which the polynomial is evaluated.
#
# OUTPUT
# [y,dydx,d2ydx2,alpha,R]=trvalues(p,x) returns the value y of the
# polynomial at x, the derivative dydx and the second derivative d2ydx2 in
# that point, as well as the slope alpha(x) and the radius of the
# osculating circle.
# The slope angle alpha is positive for a curve with a negative derivative.
# The sign of the radius of the osculating circle is the same as that of
# the second derivative.


def trvalues(p,x):
	y=np.polyval(p,x)
	dp=np.polyder(p)
	dydx=np.polyval(dp,x)
	ddp=np.polyder(dp)
	d2ydx2=np.polyval(ddp,x)
	alpha=np.arctan(-dydx)
	R=(1.0+dydx**2)**1.5/d2ydx2
	return [y,dydx,d2ydx2,alpha,R]



#, innhenter x og y verdier og plotter
# iptrack - interpolate track
#
# SYNTAX
# p=iptrack(filename)
#
# INPUT
# filename: data file containing exported tracking data on the standard
# Tracker export format
#
# mass_A
# t	x	y
# 0.0	-1.0686477620876644	42.80071293284619
# 0.04	-0.714777136706708	42.62727536827738
# ...
#
# OUTPUT
# p=iptrack(filename) returns the coefficients of a polynomial of degree 15
# that is the least square fit to the data y(x). Coefficients are given in
# descending powers.


def iptrack(filename):
	#ta bare den første, skal bare ha et utrykk for banen
	data = np.loadtxt(filename, skiprows=2)
	return np.polyfit(data[:, 1], data[:, 2], 15)


p = iptrack(r'C:\Users\Elise\Documents\6. semester\Fysikk\New\liten ball\liten ball\86.txt')


def x_positions (filename):
	x = []
	#for line in filename:
	data = np.loadtxt(filename, skiprows=2)
	for liste in data:
		x.append(liste[1])
	return(x)


x = x_positions(r'C:\Users\Elise\Documents\6. semester\Fysikk\New\liten ball\liten ball\86.txt')


def trvalues(p,x):
	# alle verdiene vi har, alle de forskjellige x-verdiene, får ut yverdiene
	#diffligning, a.ytrykk, a =v', s=v', starter med x0, plusser på xsize
	#xold, xnew, fyller
	# 1/1000
	#finner opp x-verdier, funksjonen vil gi til akurat de punktet vi putter inn
	y = np.polyval(p, x)
	dp = np.polyder(p)
	dydx = np.polyval(dp, x)
	ddp = np.polyder(dp)
	d2ydx2 = np.polyval(ddp, x)
	alpha = np.arctan(-dydx)
	R =(1.0+dydx**2)**1.5/d2ydx2
	return [y, dydx, d2ydx2, alpha, R]


x2 = [-0.4, 0.2]

#print(trvalues(p, x2)[1])


yposition_list = trvalues(p, x2)[0]
dydx_list = trvalues(p, x2)[1]
d2ydx2_list = trvalues(p, x2)[2]
alpha_list = trvalues(p, x2)[3]
R_list = trvalues(p, x2)[4]


def plotPolynomial(p):
	funksjon = ''
	ekspon = 15
	for i in p:
		funksjon += p[1]
		funksjon += x**ekspon
		ekspon -= 1
	print(funksjon)


#print(plotPolynomial(p))
r=[-0.5, 1, 0.5]
plt.plot('2r**2 + 2')
plt.show()


