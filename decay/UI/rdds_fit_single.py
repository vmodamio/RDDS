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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from time import perf_counter



tauguess = 20.0
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
upperL = 101
#mesh = np.linspace(0,upperL,4) ## define the internal knots, lower and upper limit
mesh = np.append([0.0], np.geomspace(4,upperL,4)) ## define the internal knots, lower and upper limit
ftol = 1e-8
#mesh = np.array([0,100,500,1000,3000])
## note: that the real number of splines (basis) is mesh + order -2 (6 in this case, like the weights)
# construct the A matrix

## DATA
class decay():
    instances = []
    def __init__(self, filename , order , mesh , fratio, fratio_err ):
        self.fname = filename
        self.fratio = fratio
        self.fratio_err = fratio_err
        self.__class__.instances.append(self)

        with open(self.fname, 'r') as reader:
            xx, yy, xerr, yerr = [], [], [], []
            for line in reader.readlines():
                if line.lstrip(' ')[0] == '#':  # is comment
                    continue
                else:
                    data = line.split()
                    match len(data):
                        case 2:
                            xx.append(float(data[0]))
                            xerr.append(0.05*float(data[0]))
                            yy.append(float(data[1]))
                            yerr.append(0.02/self.fratio*float(data[1]))
                        case 3:
                            xx.append(float(data[0]))
                            xerr.append(0.05*float(data[0]))
                            yy.append(float(data[1]))
                            yerr.append(float(data[2]))
                        case _:
                            xx.append(float(data[0]))
                            xerr.append(float(data[1]))
                            yy.append(float(data[2]))
                            yerr.append(float(data[3]))

        self.x = np.array(xx)
        self.y = np.array(yy)
        self.yerr = np.array(yerr)
        self.xerr = np.array(xerr)

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
    J[nx1+nx2:2*nx1+nx2, 2*n] = np.transpose(-R.dA.dot(x[:n]))
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


class rdds_fit_single():
    def __init__(self,  spline_order=3, spline_knots=3, tauguess=5.0):
        self.tauguess = tauguess
        self.spline_order = spline_order
        self.spline_knots = spline_knots
        self.spline_base = self.spline_order + self.spline_knots -2
        self.R = None
        self.F = None
        self.upperL = 80
        self.mesh_log_first = 3
        self.mesh_mode = 'log'
        self.mesh = None
        self.setMesh()
        self.ftol = 1e-8
        self.bounds = None
        self.guess = None
        self.minout = None
        self.tau_result = None
        self.tau_error = None
        self.fit_success = False
        self.fit_time = 0.0
        #self.figure = Figure(figsize=(7,10), layout='constrained', facecolor="#f0f0f0") #gridspec_kw={'height_ratios':[3, 3, 3, 2]}
        #self.figure = Figure(figsize=(7,10),layout='constrained', facecolor="#f1f1f0") 
        self.figure = Figure(figsize=(7,10),layout='constrained', facecolor="#f3f2f0") 
        self.gs = gridspec.GridSpec(nrows=4, ncols=1, figure=self.figure, height_ratios=[3,3,3,2], hspace=0)
        self.figure.get_layout_engine().set(h_pad= 0, w_pad=0, wspace=0, hspace=0, rect=(0.02, 0.02, 0.96, 0.96))
        #self.figure.tight_layout(pad=0.0, rect=(0.03, 0.03, 0.97, 0.97))
        self.ax_fed = self.figure.add_subplot(self.gs[0,0])
        self.ax_dec = self.figure.add_subplot(self.gs[1,0])
        self.ax_cur = self.figure.add_subplot(self.gs[2,0])
        self.ax_tau = self.figure.add_subplot(self.gs[3,0])
        self.ax_fed.set_facecolor('#fffffa')
        self.ax_dec.set_facecolor('#fffffa')
        self.ax_cur.set_facecolor('#fffffa')
        self.ax_tau.set_facecolor('#fffffa')
        self.mesh_fed_drawing = []
        self.mesh_dec_drawing = []
        self.canvas = FigureCanvas(self.figure)

    def setMesh(self, arr=None):
        if self.mesh_mode == 'log':
            self.mesh = np.append([0.0] , np.geomspace(self.mesh_log_first, self.upperL, self.spline_knots -1) )
        elif self.mesh_mode == 'lin':
            self.mesh = np.linspace(0.0, self.upperL, self.spline_knots)
        elif isinstance(arr, np.ndarray):
            self.mesh = arr.flatten()[:self.spline_knots]

    def setMeshMode(self, mode):
        self.mesh_mode = mode
    def setBoundsAndGuess(self):
        if isinstance(self.R, decay) and isinstance(self.F, decay):
            self.bounds = [[0.0 , None]]*(self.R.isplines.n +self.F.isplines.n  )
            self.bounds.append([0.05,None])
            self.guess = [0.1]*(self.R.isplines.n + self.F.isplines.n)
            self.guess.append(self.tauguess)

    def setDecay(self, decay):
        self.R = decay
        self.R.yerr = 4*self.R.yerr
        #self.upperL = self.R.x.max()+2
        #self.setMesh()
        self.setBoundsAndGuess()
    def setFeeder(self, decay):
        self.F = decay
        self.F.yerr = 4*self.F.yerr
        #self.upperL = self.F.x.max()+2
        #self.setMesh()
        self.setBoundsAndGuess()

    def runFit(self):
        t0 = perf_counter()
        self.minout = minimize(fnm, self.guess, args=(self.R, self.F), method='SLSQP', 
                               jac= '2-point', bounds = self.bounds, options={'disp': True, 'ftol': self.ftol})

        self.fit_time = perf_counter() - t0

        self.R.weights = self.minout.x[:self.R.isplines.n]
        self.F.weights = self.minout.x[self.R.isplines.n : self.R.isplines.n + self.F.isplines.n]

    def clearMesh(self):
        for line in self.mesh_fed_drawing:
            if isinstance(line, Line2D):
                line.remove()
        for line in self.mesh_dec_drawing:
            if isinstance(line, Line2D):
                line.remove()

    def drawMesh(self):

        self.mesh_fed_drawing = [self.ax_fed.axvline(x=knot, color='#878787', linestyle='--', linewidth=1) for knot in self.mesh]
        self.mesh_dec_drawing = [self.ax_dec.axvline(x=knot, color='#878787', linestyle='--', linewidth=1) for knot in self.mesh]
        self.canvas.draw()
        #self.ax_fed.axvline(x=self.mesh, color='#878787', linestyle='--', linewidth=1) 
        #self.ax_dec.axvline(x=self.mesh, color='#878787', linestyle='--', linewidth=1) 
        #self.mesh_dec_drawing.set_xdata([knot,knot])
        #self.mesh_dec_drawing[0].set_visible(False)

    def drawFigure(self):
        self.ax_fed.clear()
        self.ax_dec.clear()
        self.ax_cur.clear()
        self.ax_tau.clear()
        xgrid = np.linspace(0.0, self.upperL, 3000, endpoint=False)
        splines = Isplines_total(self.spline_order, self.mesh, xgrid)
        splbasis = Isplines(self.spline_order, self.mesh, xgrid)
        msplbasis = Msplines(self.spline_order, self.mesh, xgrid)
        
        #plt.rcParams["figure.figsize"] = (7,10)
        #self.figure, (ax_fed, ax_dec, ax_cur, ax_tau) = plt.subplots(4,1, gridspec_kw={'height_ratios':[3, 3, 3, 2]})
        self.drawMesh()
        #for knot in self.mesh:
        #    self.ax_fed.axvline(x=knot, color='#878787', linestyle='--', linewidth=1)
        #    self.ax_dec.axvline(x=knot, color='#878787', linestyle='--', linewidth=1)
        
        self.ax_fed.plot(xgrid, np.exp(-splines.Itotal(self.F.weights,0)), color='orangered')
        self.ax_fed.errorbar(self.F.x, self.F.y, xerr= self.F.xerr, yerr= self.F.yerr, 
                        fmt='o', markersize=2, capsize=3, linewidth=1, color='black')
        
        self.ax_dec.plot(xgrid, np.exp(-splines.Itotal(self.R.weights,0)), color='orangered')
        self.ax_dec.errorbar(self.R.x, self.R.y, xerr= self.R.xerr, yerr= self.R.yerr, 
                        fmt='o', markersize=2, capsize=3, linewidth=1, color='black')
        
        deltaR =  np.exp(-splines.Itotal(self.R.weights,0)) - self.F.fratio*np.exp(-splines.Itotal(self.F.weights,0))
        self.ax_cur.plot(xgrid, deltaR, color='grey', label=r'$R(t)-\eta_i\, F_i(t)$')
        
        self.ax_cur.plot(xgrid, self.minout.x[-1]*splines.dItotal_dx(self.R.weights)*np.exp(-splines.Itotal(self.R.weights,0)), 
                    color='blue', label=r'$-\tau\,\frac{\partial R(t)}{\partial t}$')
        self.ax_cur.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.97,0.94), fontsize=14)
        
        taus = deltaR/(splines.dItotal_dx(self.R.weights)*np.exp(-splines.Itotal(self.R.weights,0)) )/ve
        
        discretesplines = Isplines_total(self.spline_order, self.mesh, self.R.x)
        discretedeltaR =  np.exp(-discretesplines.Itotal(self.R.weights,0)) - self.F.fratio*np.exp(-discretesplines.Itotal(self.F.weights,0))
        # TAUS With the experimental data, except for the slope
        #discretedeltaR = (1.0 -F.fratio) - R.y +F.fratio*F.y
        discretetaus = discretedeltaR/(discretesplines.dItotal_dx(self.R.weights)*np.exp(-discretesplines.Itotal(self.R.weights,0)) )/ve
        
        ## Error estimates for curves
        wval = self.minout.x[:self.R.isplines.n]
        vval = self.minout.x[self.R.isplines.n: 2*self.R.isplines.n]
        tval = self.minout.x[-1]
        self.tau_result = tval
        werr = weighterrors(self.minout.x, self.R,self.F)[:self.R.isplines.n]
        verr = weighterrors(self.minout.x, self.R,self.F)[self.R.isplines.n: 2*self.R.isplines.n]
        terr = weighterrors(self.minout.x, self.R,self.F)[-1]
        self.tau_error = terr
        dRerr2 = self.R.dA**2 @ werr**2 
        Rerr2  = self.R.A**2 @ werr**2 
        Ferr2  = self.F.fratio**2*(self.R.A**2 @ verr**2)
        dRval2 = (self.R.dA @ self.R.weights)**2 
        
        discreteR = discretesplines.Itotal(self.R.weights,0)
        dRval = self.R.dA @ self.R.weights 
        G = np.diag(1./dRval)
        H = np.diag(discretedeltaR/dRval**2)
        dtdw2 = (G@self.R.A +H@self.R.dA)**2 @ werr**2
        dtdv2 = (self.F.fratio*G@self.R.A)**2 @ verr**2
        te = np.sqrt(dtdw2) # + dtdv2)
        #print(te)
        print(f'Effective ndf: {effndf(self.minout.x, self.R,self.F)}')
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
        tauerr = discretetaus*np.sqrt((self.R.yerr**2 +self.F.fratio**2*self.R.Fyerr**2)/discretedeltaR**2  +dRerr2/dRval2)
        tauerrW = terr*np.sqrt(np.sum(1.0/tauerr**2))
        tauerrN = discretetaus*tauerrW*np.sqrt((self.R.yerr**2 +self.F.fratio**2*self.R.Fyerr**2)/discretedeltaR**2  +dRerr2/dRval2)
        print('-- Distance    Lifetime    Err   Discrete values --------')
        for dist, lifetime, error in zip(self.R.x,discretetaus,tauerrN):
            print('{0:8.2f}    {1:8.3f}    {2:8.3f}'.format(dist,lifetime,error))
        
        self.ax_tau.plot(xgrid, taus, color='goldenrod', linewidth=1)
        self.ax_tau.scatter(self.R.x, discretetaus, color='black', s=10)
        self.ax_tau.errorbar(self.R.x, discretetaus, yerr= tauerrN, fmt='o', markersize=2, capsize=3, linewidth=1, color='black')
        self.ax_tau.axhline(y=self.minout.x[-1], color='#878787', linestyle='--', linewidth=1)
        
        self.ax_fed.tick_params(axis="both", direction="in", length=6)
        self.ax_dec.tick_params(axis="both", direction="in", length=6)
        self.ax_cur.tick_params(axis="both", direction="in", length=6)
        self.ax_tau.tick_params(axis="both", direction="in", length=6)
        
        self.ax_tau.set_xlabel(r'Time ($ps$)', labelpad=8)
        self.ax_tau.set_ylabel(r'$\tau\; (ps)$')
        self.ax_fed.set_ylabel(r'$F_{i}(t)$')
        self.ax_dec.set_ylabel(r'$R(t)$')
        self.ax_cur.set_ylabel(r'$-\tau\,({\partial R}/{\partial t})$')
        
        #plt.setp(ax_fed.get_xticklabels(), visible=False)
        #plt.setp(ax_dec.get_xticklabels(), visible=False)
        #plt.setp(ax_cur.get_xticklabels(), visible=False)
        self.ax_fed.set_xticklabels([])
        self.ax_dec.set_xticklabels([])
        self.ax_cur.set_xticklabels([])
        
        #self.figure.tight_layout(pad=0.0, rect=(0.03, 0.03, 0.97, 0.97))
        self.canvas.draw()

