""" From multifit_..._NEGATIVE_truncated_base. 
So far the constraints in the I-spline coefficients were (0,1) for each coeff. 
in both decay curves, plus (0.05, inf) for the lifetime. Written as Bounds.
Additionally, a linear constraint set so the sum of all coeffs. for a decay curve
lies within (0,1). So, there wasnt a extrict normalization to 1.0 actually.
With all this, the final I-spline would start at 0 for the frist knot, and end at 
a value of 1.0 at the last knot (or less, actually, will end it at sum_coeffs).
This was kind of hidden, as I was using monotone increasing curves, starting from 0.

When changing the program to work with R(t), decreasing monotone curves. The 
normalization factors pop up! For example, in the main decay curve for the 100Zr, the 
last knot being around 0.2, would make the fit such the sum is ~0.8, if you choose to
settle the last knot around the last experimental point.

This would produce random no-convergences in the fit, depending of the proximity of that
knot to the point, and produce strange curves at the end. Specially because the last 
spline in the base is curved the opposite! 

Constraining the sum_coeffs to actually 1.0, and choosing the last knot relatively far 
from the last exp. point, would converge (this is on the multifit_NEGATIVE last attempt).

Tried also to truncate the base to remove the last spline, in order to check whether the 
asympthotic behaviour improves, but it makes a mesh.

LOG-R fitting now: the curve is parameterized as -logR = I-splines!! , with monotone increasing
splines.
So now R = exp(-Sum[w_i I_i]) and is explicitly normalized to 1.0 as far as the coeffs are 
positive. 
No need to constraint the sum, and no need to limit the coeffs within (0,1). They just need to
be positive.

"""
import itertools
import math
import numpy as np
import pandas as pd
import scipy.optimize
from ispline import Isplines_total, Isplines, Msplines
from scipy.optimize import nnls
from scipy.optimize import SR1
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import cholesky
#import matplotlib
import matplotlib.pyplot as plt

#from matplotlib import rc
#plt.rcParams["text.usetex"] = True
#rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## DATA
xT = np.array([ 30, 52, 89, 155, 264, 450, 776, 1170, 1776, 2651 ])
yR  = 1.0 - np.exp(-xT/750.0)
yRf = 1.0 - np.exp(-xT/350.0)
feeding = 0.6
yRt = (yR +feeding*yRf)/(1.0+feeding)
#yR = np.array([0.05, 0.15, 0.42, 0.49, 0.54, 0.48, 0.73, 0.83, 0.78, 0.89])

## for testing purposes, generate a R curve with decay time distance 700, and feeder 350 with feeding 0.7
#testF = np.exp(-xT/350.0)
#testR = 0.3*np.exp(-xT/700.0) +testF + 0.7*700.0/(350.0-700.0)*(np.exp(-xT/350.0) -np.exp(-xT/700.0))
#for i,dis in enumerate(xT):
#    print(f'{dis}   {10000.0*testR[i]}   100.0   {10000.0*(1.0-testR[i])}    100.0')
#for i,dis in enumerate(xT):
#    print(f'{dis}   {10000.0*testF[i]}   100.0   {10000.0*(1.0-testF[i])}    100.0')

tauguess = 5.0
terr = 1.0
clight = 299.79 # um/ps
beta0 = 0.103
ve = beta0*clight
ve = 1.0
## Contruct the spline basis 
#order = 3
#upperL = 72.85
order = 4
#upperL = 72.9
upperL = 73
#mesh = np.linspace(0,upperL,4) ## define the internal knots, lower and upper limit
mesh = np.append([0.0], np.geomspace(4,upperL,4)) ## define the internal knots, lower and upper limit
ftol = 1e-8
#mesh = np.array([0,100,500,1000,3000])
## note: that the real number of splines (basis) is mesh + order -2 (6 in this case, like the weights)
# construct the A matrix

## DATA
class decay():
    instances = []
    def __init__(self, filename , state , feeds , fratio ):
        self.fname = filename
        self.state = state
        self.feeds = feeds
        self.fratio = fratio
        self.__class__.instances.append(self)

        with open(self.fname, 'r') as reader:
            xx, yy, xerr, yerr = [], [], [], []
            for line in reader.readlines():
                data = line.split()
                xx.append(float(data[0]))
                s = float(data[3])
                se = float(data[4])
                u = float(data[1])
                ue = float(data[2])
                yy.append(u/(u+s))
                e1 = ue*(s/(u+s)/(u+s))
                e2 = se*(u/(u+s)/(u+s))
                yerr.append(math.sqrt(e1*e1+e2*e2))

        self.x = np.array(xx)
        self.y = np.array(yy)
        self.yerr = np.array(yerr)
        self.xerr = 0.1

        self.Fyerr = np.array([])

        self.isplines = Isplines(order, mesh, self.x)
        #self.isplines.lower = 0.0
        #self.isplines.upper = 0.0
        self.A = np.array([self.isplines.I(i) for i in range(1, self.isplines.n + 1)]) 
        self.A = np.transpose(self.A)
        self.dA = np.array([self.isplines.dI_dx(i) for i in range(1, self.isplines.n + 1)]) 
        self.dA = np.transpose(self.dA)
        self.weights, self.residual = nnls(self.A, self.y)

def fn(x, A, b):
    return np.linalg.norm(A.dot(x)-b)


#def fnm(x, R, F):
#    n = R.isplines.n
#    c1 = (R.A.dot(x[:n])-R.y)/R.yerr
#    c2 = (F.A.dot(x[n: 2*n])-F.y)/F.yerr 
#    c3 = (x[2*n]*R.dA.dot(x[:n]) +R.A.dot(x[:n] -F.fratio*x[n: 2*n]) -(1.0 -F.fratio))/terr
#    return np.linalg.norm(c1) + np.linalg.norm(c2) + np.linalg.norm(c3)

def Res(x, R, F):
    nx1 = len(R.x)
    nx2 = len(F.x)
    n = R.isplines.n
    Rs = np.zeros(2*nx1+nx2)
    Rs[:nx1] = (np.exp(-R.A.dot(x[:n])) -R.y)
    Rs[nx1:nx1+nx2] = (np.exp(-F.A.dot(x[n: 2*n])) -F.y)
    Rs[nx1+nx2:] = 1.0 - x[2*n]*R.dA.dot(x[:n]) -F.fratio*np.exp( R.A.dot(x[:n] -x[n: 2*n]) ) 
    return np.transpose(Rs)

def rW(R,F):
    nx1 = len(R.x)
    nx2 = len(F.x)
    d = np.zeros(2*nx1+nx2)
    d[:nx1] = R.yerr**2
    d[nx1:nx1+nx2] = F.yerr**2

    # d[nx1+nx2:2*nx1+nx2] = R.yerr**2 +F.fratio**2*F.yerr**2 ## here problem: as R and F does not necesarily have same X's
    R.Fyerr = np.array([])
    for rx in R.x:
        mid = np.argmin((F.x -rx)**2)
        R.Fyerr = np.append( R.Fyerr, F.yerr[mid] )

    d[nx1+nx2:2*nx1+nx2] = R.yerr**2 +F.fratio**2*R.Fyerr**2 ## here problem: as R and F does not necesarily have same X's
    W = np.linalg.inv(np.diag(d))
    C = cholesky(W)
    return C.T

def fnm(x, R, F):
    #return 0.5*np.transpose(Res(x,R,F)) @ Res(x,R,F)
    WR = rW(R,F) @ Res(x,R,F)
    return 0.5*np.transpose(WR) @ WR

def Xmat(x, R, F):
    nx1 = len(R.x)
    nx2 = len(F.x)
    n = R.isplines.n
    M = np.zeros((2*nx1+nx2, 2*n))
    M[:nx1,:n] = np.transpose(np.array([R.isplines.I(i) for i in range(1, R.isplines.n + 1)])) 
    M[nx1:nx1+nx2,n:2*n] = np.transpose(np.array([F.isplines.I(i) for i in range(1, F.isplines.n + 1)])) 
    M[nx1+nx2:2*nx1+nx2,:n] = np.transpose(np.array([x[2*n]*R.isplines.dI_dx(i) +R.isplines.I(i) for i in range(1, R.isplines.n + 1)]))
    M[nx1+nx2:2*nx1+nx2,n:2*n] = np.transpose(np.array([-F.fratio*R.isplines.I(i) for i in range(1, R.isplines.n + 1)])) 
    return M

def effndf(x,R,F):
    X = Xmat(x,R,F)
    W = rW(R,F) @ rW(R,F)
    hat = X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W
    hatd = np.diagonal(hat)
    return hatd.sum()

def Jac(x, R, F):
    nx1 = len(R.x)
    nx2 = len(F.x)
    n = R.isplines.n
    J = np.zeros((2*nx1+nx2, 2*n+1))
    J[:nx1,:n] = np.transpose(-np.exp(-R.A.dot(x[:n])) * np.array([R.isplines.I(i) for i in range(1, R.isplines.n + 1)])) 
    J[nx1:nx1+nx2,n:2*n] = np.transpose(-np.exp(-F.A.dot(x[n:2*n])) * np.array([F.isplines.I(i) for i in range(1, F.isplines.n + 1)])) 
    fexp = -F.fratio*np.exp( R.A.dot(x[:n] -x[n: 2*n]) )
    di = np.array([-x[2*n]*R.isplines.dI_dx(i) for i in range(1, R.isplines.n + 1)])
    dexp = np.array([R.isplines.I(i) for i in range(1, R.isplines.n + 1)])
    J[nx1+nx2:2*nx1+nx2,:n] = np.transpose(di +fexp*dexp)
    J[nx1+nx2:2*nx1+nx2,n:2*n] = np.transpose(-fexp*dexp) 
    J[nx1+nx2:2*nx1+nx2, 2*n] = -np.transpose(R.dA.dot(x[:n]))
    #return J
    return rW(R,F) @ J

def gradient(x,R,F):
    #return np.transpose(Jac(x,R,F)) @ Res(x,R,F)
    WR = rW(R,F) @ Res(x,R,F)
    return np.transpose(Jac(x,R,F)) @ WR

def d2Q(x, R, F):
    n = R.isplines.n
    nx1 = len(R.x)
    nx2 = len(F.x)
    Q = np.zeros((2*n+1, 2*n+1))
    WR = rW(R,F) @ Res(x,R,F)
    #Q[2*n:2*n+1, :n] = np.array([ Res(x,R,F)[:nx1] @ R.isplines.dI_dx(i) for i in range(1, n+1)])
    ## Here, the sum expands only to the last residuals, only those wich second derivative is not null
    Q[2*n:2*n+1, :n] = np.array([ WR[nx1+nx2:] @ R.isplines.dI_dx(i) for i in range(1, n+1)])
    Q[:n, 2*n:2*n+1] = np.transpose(Q[2*n:2*n+1, :n])
    return Q

def hess(x,R,F):
    return np.transpose(Jac(x,R,F)) @ Jac(x,R,F) + d2Q(x,R,F)

def weighterrors(x,R,F):
    nx1 = len(R.x)
    nx2 = len(F.x)
    n = R.isplines.n
    #ndf = nx1 + nx2 - (2*n+1)
    ndf = effndf(x,R,F)
    WR = rW(R,F) @ Res(x,R,F)
    #WR = Res(x,R,F)
    #S = np.sqrt(np.transpose(Res(x,R,F)) @ Res(x,R,F) /ndf)
    S = (np.transpose(WR) @ WR) /ndf
    #S = (np.transpose(WR) @ WR) 
    sensitivity = np.linalg.inv(hess(x,R,F))
    return np.sqrt(S*np.diag(sensitivity))

def dRerrors(x,R,F):
    nx1 = len(R.x)
    n = R.isplines.n
    ndf = nx1 - n
    WR = rW(R,F) @ Res(x,R,F)
    S = (np.transpose(WR)[:nx1] @ WR[:nx1]) /ndf
    sensitivity = np.linalg.inv(hess(x,R,F))
    werr = np.sqrt(S*np.diag(sensitivity))
    dRerr2 = R.dA**2 @ werr[:n]**2 
    dRval2 = R.dA**2 @ R.weights**2 
    return np.sqrt(dRerr2/dRval2)

#R = decay('testr.dat', 4, 2, 1.0)
#F = decay('testf.dat', 6, 4, 0.7)

R = decay('zr100_4plus.dat', 4, 2, 1.0)
F = decay('zr100_6plus.dat', 6, 4, 0.83)

## emulate no errors...
#R.yerr = 0.0*R.yerr +1.0
#F.yerr = 0.0*F.yerr +1.0

## Add random noise to the data points:
#R.y = np.random.normal(R.y, 0.02)
#F.y = np.random.normal(F.y, 0.02)

# this is an approach because the errors in the decay class are wrongly small.
R.yerr = 4*R.yerr 
F.yerr = 4*F.yerr 

# discrete random
#R.y = np.random.normal(R.y, R.yerr)
#F.y = np.random.normal(F.y, F.yerr)

bounds = [[0.0 , None]]*(R.isplines.n +F.isplines.n  )
# remove the contribution from the last spline
#bounds[R.isplines.n-1] = [0.0, 0.0]
#bounds[R.isplines.n +F.isplines.n -1] = [0.0, 0.0]
bounds.append([0.05,None])
#x0 = [0.1 for i in range(1,F.isplines.n+1)]
x0 = [0.1]*(R.isplines.n +F.isplines.n)
x0.append(tauguess)

## MINIMIZATION
#minout = minimize(fnm, x0, args=(R, F), method='SLSQP', jac= gradient, bounds = bounds, options={'disp': True, 'ftol': ftol})
minout = minimize(fnm, x0, args=(R, F), method='SLSQP', jac= '2-point', bounds = bounds, options={'disp': True, 'ftol': ftol})
#minout = minimize(fnm, x0, args=(R, F), method='SLSQP', jac= '2-point', bounds = bounds, options={'disp': True, 'ftol': ftol})
R.weights = minout.x[:R.isplines.n]
F.weights = minout.x[R.isplines.n : R.isplines.n +F.isplines.n]

#Rerr = np.sqrt(ftol*np.diag(minout.hess_inv.todense())[:R.isplines.n])
#print(minout.hess_inv)


#print('SLSQP spline weights: ')
#print(minout.x)

#print('--------------------------')
#print(weighterrors(minout.x, R,F))
#print('--------------------------')

xgrid = np.linspace(0.0, upperL, 3000, endpoint=False)
splines = Isplines_total(order, mesh, xgrid)
splbasis = Isplines(order, mesh, xgrid)
msplbasis = Msplines(order, mesh, xgrid)

plt.rcParams["figure.figsize"] = (7,10)
fig, (ax_fed, ax_dec, ax_cur, ax_tau) = plt.subplots(4,1, gridspec_kw={'height_ratios':[3, 3, 3, 2]})

for knot in mesh:
    ax_fed.axvline(x=knot, color='#878787', linestyle='--', linewidth=1)
    ax_dec.axvline(x=knot, color='#878787', linestyle='--', linewidth=1)

ax_fed.plot(xgrid, np.exp(-splines.Itotal(F.weights,0)), color='orangered')
ax_fed.errorbar(F.x, F.y, xerr= F.xerr, yerr= F.yerr, fmt='o', markersize=2, capsize=3, linewidth=1, color='black')

ax_dec.plot(xgrid, np.exp(-splines.Itotal(R.weights,0)), color='orangered')
ax_dec.errorbar(R.x, R.y, xerr= R.xerr, yerr= R.yerr, fmt='o', markersize=2, capsize=3, linewidth=1, color='black')

deltaR =  np.exp(-splines.Itotal(R.weights,0)) -F.fratio*np.exp(-splines.Itotal(F.weights,0))
ax_cur.plot(xgrid, deltaR, color='grey', label=r'$R(t)-\eta_i\, F_i(t)$')

ax_cur.plot(xgrid, minout.x[-1]*splines.dItotal_dx(R.weights)*np.exp(-splines.Itotal(R.weights,0)), color='blue', 
            label=r'$-\tau\,\frac{\partial R(t)}{\partial t}$')
ax_cur.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.97,0.94), fontsize=14)

taus = deltaR/(splines.dItotal_dx(R.weights)*np.exp(-splines.Itotal(R.weights,0)) )/ve

discretesplines = Isplines_total(order, mesh, R.x)
discretedeltaR =  np.exp(-discretesplines.Itotal(R.weights,0)) -F.fratio*np.exp(-discretesplines.Itotal(F.weights,0))
# TAUS With the experimental data, except for the slope
#discretedeltaR = (1.0 -F.fratio) - R.y +F.fratio*F.y
discretetaus = discretedeltaR/(discretesplines.dItotal_dx(R.weights)*np.exp(-discretesplines.Itotal(R.weights,0)) )/ve

## Error estimates for curves
wval = minout.x[:R.isplines.n]
vval = minout.x[R.isplines.n: 2*R.isplines.n]
tval = minout.x[-1]
werr = weighterrors(minout.x, R,F)[:R.isplines.n]
verr = weighterrors(minout.x, R,F)[R.isplines.n: 2*R.isplines.n]
terr = weighterrors(minout.x, R,F)[-1]
dRerr2 = R.dA**2 @ werr**2 
Rerr2  = R.A**2 @ werr**2 
Ferr2  = F.fratio**2*(R.A**2 @ verr**2)
dRval2 = (R.dA @ R.weights)**2 

discreteR = discretesplines.Itotal(R.weights,0)
dRval = R.dA @ R.weights 
G = np.diag(1./dRval)
H = np.diag(discretedeltaR/dRval**2)
dtdw2 = (G@R.A +H@R.dA)**2 @ werr**2
dtdv2 = (F.fratio*G@R.A)**2 @ verr**2
te = np.sqrt(dtdw2) # + dtdv2)
#print(te)

print(f'Effective ndf: {effndf(minout.x, R,F)}')

print('-- Spline Coeffs. Main decay -----')
for parameter, error in zip(wval,werr):
    print('{0:10.6f}    {1:10.6f}'.format(parameter,error))
print(f"SUM: {'{0:8.6f}'.format(wval.sum())}")
print('-- Spline Coeffs. Feeding decay -----')
for parameter, error in zip(vval,verr):
    print('{0:10.6f}    {1:10.6f}'.format(parameter,error))
print(f"SUM: {'{0:8.6f}'.format(vval.sum())}")
print('-- Main decay [Lifetime] -----')
print('{0:10.4f}    {1:10.4f}'.format(tval,terr))

## using exp errors...
tauerr = discretetaus*np.sqrt((R.yerr**2 +F.fratio**2*R.Fyerr**2)/discretedeltaR**2  +dRerr2/dRval2)
tauerrW = terr*np.sqrt(np.sum(1.0/tauerr**2))
tauerrN = discretetaus*tauerrW*np.sqrt((R.yerr**2 +F.fratio**2*R.Fyerr**2)/discretedeltaR**2  +dRerr2/dRval2)
print('-- Distance    Lifetime    Err   Discrete values --------')
for dist, lifetime, error in zip(R.x,discretetaus,tauerrN):
    print('{0:8.2f}    {1:8.3f}    {2:8.3f}'.format(dist,lifetime,error))

#toterr = np.sqrt(1./np.sum(1./tauerrN**2))
#print(f'total err estimate is {toterr}')


ax_tau.plot(xgrid, taus, color='goldenrod', linewidth=1)
ax_tau.scatter(R.x, discretetaus, color='black', s=10)
ax_tau.errorbar(R.x, discretetaus, yerr= tauerrN, fmt='o', markersize=2, capsize=3, linewidth=1, color='black')
ax_tau.axhline(y=minout.x[-1], color='#878787', linestyle='--', linewidth=1)

ax_fed.tick_params(axis="both", direction="in", length=6)
ax_dec.tick_params(axis="both", direction="in", length=6)
ax_cur.tick_params(axis="both", direction="in", length=6)
ax_tau.tick_params(axis="both", direction="in", length=6)

ax_tau.set_xlabel(r'Time ($ps$)', labelpad=8)
ax_tau.set_ylabel(r'$\tau\; (ps)$')
ax_fed.set_ylabel(r'$F_{i}(t)$')
ax_dec.set_ylabel(r'$R(t)$')
ax_cur.set_ylabel(r'$-\tau\,({\partial R}/{\partial t})$')

plt.setp(ax_fed.get_xticklabels(), visible=False)
plt.setp(ax_dec.get_xticklabels(), visible=False)
plt.setp(ax_cur.get_xticklabels(), visible=False)

fig.tight_layout(pad=0.0, rect=(0.03, 0.03, 0.97, 0.97))
plt.show()

## Notes on errors: still need to propagate the error in beta/time, and the error in the feeding

xslices = np.array( [1.20122 , 
                     1.77571 , 
                     2.6614  , 2.94228 , 
                     4.23931 , 4.54704 , 4.86336 , 
                     6.88887 , 7.2994  , 7.84516 , 
                     11.3299 , 11.902  , 12.4944 , 13.2391 , 
                     19.327  , 20.205  , 21.0428 , 22.3813 , 
                     29.1077 , 30.5513 , 31.9798 , 33.8729 , 
                     43.9271 , 46.1506 , 48.2787 , 51.0786 , 
                     65.5834 , 69.1406 , 72.453  , 76.4397 ])

splslices = Isplines_total(order, mesh, xslices)
slices = splslices.Itotal(R.weights,0) 
for sl in slices:
    print(sl)
