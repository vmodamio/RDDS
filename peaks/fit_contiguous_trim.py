"""
Step by step checking for new technique
0) First, make a simple fit to gaussians plus linear bkg to start
1) Background: mask the area from u0-3*s0 to u1+3*s1 and fit the 
   background with a soft1L los function to avoid outliers.
   Note: outliers from contamination in the background, and from the
   fact that the previous gaussians are not the definitive.
   The scaling factor could be retrieved from the background fit from 
   the previous point, and the rms.

"""
import sys
#import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import least_squares
from scipy import stats

#parser = argparse.ArgumentParser(description='Fit beta-slices with discrete degrader')
#parser.add_argument('Distance N', metavar='N', type=int, nargs=1,
#                    help='Distance to be fitted')
#parser.add_argument('--bound', dest='accumulate', action='store_const',
#                    const=sum, default=False,
#                    help='Bound fit parameters to previous fitted distance')
#
#args = parser.parse_args()
#print(args.accumulate(args.integers))


## fit related
ftol = 1e-8
plt.rcParams["figure.figsize"] = (16,8)

## peaks to fit
xrmin = 1165
xrmax = 1265
#guess = [200.0 , 1204.0 , 3.0, 300.0, 1222, 2.0, 100.0, 0.0 ]
guess = [400.0 , 1204.0 , 5.5, 700.0, 1222, 2.5, 50.0, 0.01 ]

#fixedparameter = [{'id': 0 , 'min': 200, 'max': 200}]
fixedparameter = []

bins = 4000
xran = 2000
chwidth = xran/bins
dx = np.linspace(0.5*chwidth, 0.5*chwidth + xran, bins, endpoint=False)
degrader_thickness = 25.89 # um

trim = 10
trim_sigmas = 0.0
trim_std = np.linspace(1,trim, trim).std()

# File holding candidate bounds (from previous fit)
fbound = "prev_fit.dat"
fintegrals = "prev_int.dat"
savefit = True

boundfit = True if len(sys.argv) > 2 and sys.argv[2] == 'True' else False
print(f'Fit set with boundings {boundfit}')


if boundfit:
    ## Fitting to  Area , Centroid, Sigma, Area, Centroid, Sigma
    bound_pars = [ False, True, True,   False, True, True]
    bound_frac = [ 0.00 , 0.0004, 0.02 ,   0.00,  0.0004, 0.02 ]
    prevfit = np.loadtxt(fbound) 
    prevint = np.loadtxt(fintegrals)


#data = [ {'t0': 0.8524 , 'dt': 0.63319443, 'file': 'd30um_b0130_b0144.dat'  }, 
#         {'t0': 0.9431 , 'dt': 0.66499171, 'file': 'd30um_b0122_b0130.dat'  }, 
#         {'t0': 0.9915 , 'dt': 0.76149776, 'file': 'd30um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 1.412  , 'dt': 0.63319443, 'file': 'd52um_b0130_b0144.dat'  }, 
#         {'t0': 1.506  , 'dt': 0.66499171, 'file': 'd52um_b0122_b0130.dat'  }, 
#         {'t0': 1.615  , 'dt': 0.76149776, 'file': 'd52um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 2.363  , 'dt': 0.63319443, 'file': 'd89um_b0130_b0144.dat'  }, 
#         {'t0': 2.519  , 'dt': 0.66499171, 'file': 'd89um_b0122_b0130.dat'  }, 
#         {'t0': 2.714  , 'dt': 0.76149776, 'file': 'd89um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 3.978  , 'dt': 0.63319443, 'file': 'd155um_b0130_b0144.dat'  }, 
#         {'t0': 4.253  , 'dt': 0.66499171, 'file': 'd155um_b0122_b0130.dat'  }, 
#         {'t0': 4.571  , 'dt': 0.76149776, 'file': 'd155um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 6.686  , 'dt': 0.63319443, 'file': 'd264um_b0130_b0144.dat'  }, 
#         {'t0': 7.149  , 'dt': 0.66499171, 'file': 'd264um_b0122_b0130.dat'  }, 
#         {'t0': 7.679  , 'dt': 0.76149776, 'file': 'd264um_b0100_b0122.dat'  }]
                        
#data =  [{'t0': 11.29  , 'dt': 0.63319443 ,'file': 'd450um_b0130_b0144.dat'  },
#         {'t0': 12.06  , 'dt': 0.66499171 ,'file': 'd450um_b0122_b0130.dat'  }, 
#         {'t0': 12.96  , 'dt': 0.76149776 ,'file': 'd450um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 19.41  , 'dt': 0.63319443, 'file': 'd780um_b0130_b0144.dat'  }, 
#         {'t0': 20.75  , 'dt': 0.66499171, 'file': 'd780um_b0122_b0130.dat'  }, 
#         {'t0': 22.29  , 'dt': 0.76149776, 'file': 'd780um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 29.20  , 'dt': 0.63319443, 'file': 'd1170um_b0130_b0144.dat'  }, 
#         {'t0': 31.21  , 'dt': 0.66499171, 'file': 'd1170um_b0122_b0130.dat'  }, 
#         {'t0': 33.53  , 'dt': 0.76149776, 'file': 'd1170um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 44.25  , 'dt': 0.63319443, 'file': 'd1776um_b0130_b0144.dat'  }, 
#         {'t0': 47.31  , 'dt': 0.66499171, 'file': 'd1776um_b0122_b0130.dat'  }, 
#         {'t0': 50.83  , 'dt': 0.76149776, 'file': 'd1776um_b0100_b0122.dat'  }]
                        
#data = [ {'t0': 65.96  , 'dt': 0.63319443, 'file': 'd2650um_b0130_b0144.dat'  }, 
#         {'t0': 70.55  , 'dt': 0.66499171, 'file': 'd2650um_b0122_b0130.dat'  }, 
#         {'t0': 75.80  , 'dt': 0.76149776, 'file': 'd2650um_b0100_b0122.dat'  }]

spectra = {
        0 : [ {'t0': 0.8524 , 'dt': 0.63319443, 'file': 'd30um_b0130_b0144.dat'  }, 
                 {'t0': 0.9431 , 'dt': 0.66499171, 'file': 'd30um_b0122_b0130.dat'  }, 
                 {'t0': 0.9915 , 'dt': 0.76149776, 'file': 'd30um_b0100_b0122.dat'  }] ,
        1 : [ {'t0': 1.412  , 'dt': 0.63319443, 'file': 'd52um_b0130_b0144.dat'  }, 
                 {'t0': 1.506  , 'dt': 0.66499171, 'file': 'd52um_b0122_b0130.dat'  }, 
                 {'t0': 1.615  , 'dt': 0.76149776, 'file': 'd52um_b0100_b0122.dat'  }] ,
        2 : [ {'t0': 2.363  , 'dt': 0.63319443, 'file': 'd89um_b0130_b0144.dat'  }, 
                 {'t0': 2.519  , 'dt': 0.66499171, 'file': 'd89um_b0122_b0130.dat'  }, 
                 {'t0': 2.714  , 'dt': 0.76149776, 'file': 'd89um_b0100_b0122.dat'  }] ,
        3 : [ {'t0': 3.978  , 'dt': 0.63319443, 'file': 'd155um_b0130_b0144.dat'  }, 
                 {'t0': 4.253  , 'dt': 0.66499171, 'file': 'd155um_b0122_b0130.dat'  }, 
                 {'t0': 4.571  , 'dt': 0.76149776, 'file': 'd155um_b0100_b0122.dat'  }] ,
        4 : [ {'t0': 6.686  , 'dt': 0.63319443, 'file': 'd264um_b0130_b0144.dat'  }, 
                 {'t0': 7.149  , 'dt': 0.66499171, 'file': 'd264um_b0122_b0130.dat'  }, 
                 {'t0': 7.679  , 'dt': 0.76149776, 'file': 'd264um_b0100_b0122.dat'  }] ,
        5 : [{'t0': 11.29  , 'dt': 0.63319443 ,'file': 'd450um_b0130_b0144.dat'  },
                 {'t0': 12.06  , 'dt': 0.66499171 ,'file': 'd450um_b0122_b0130.dat'  }, 
                 {'t0': 12.96  , 'dt': 0.76149776 ,'file': 'd450um_b0100_b0122.dat'  }] ,
        6 : [ {'t0': 19.41  , 'dt': 0.63319443, 'file': 'd780um_b0130_b0144.dat'  }, 
                 {'t0': 20.75  , 'dt': 0.66499171, 'file': 'd780um_b0122_b0130.dat'  }, 
                 {'t0': 22.29  , 'dt': 0.76149776, 'file': 'd780um_b0100_b0122.dat'  }] ,
        7 : [ {'t0': 29.20  , 'dt': 0.63319443, 'file': 'd1170um_b0130_b0144.dat'  }, 
                 {'t0': 31.21  , 'dt': 0.66499171, 'file': 'd1170um_b0122_b0130.dat'  }, 
                 {'t0': 33.53  , 'dt': 0.76149776, 'file': 'd1170um_b0100_b0122.dat'  }] ,
        8 : [ {'t0': 44.25  , 'dt': 0.63319443, 'file': 'd1776um_b0130_b0144.dat'  }, 
                 {'t0': 47.31  , 'dt': 0.66499171, 'file': 'd1776um_b0122_b0130.dat'  }, 
                 {'t0': 50.83  , 'dt': 0.76149776, 'file': 'd1776um_b0100_b0122.dat'  }] ,
        9 : [ {'t0': 65.96  , 'dt': 0.63319443, 'file': 'd2650um_b0130_b0144.dat'  }, 
                 {'t0': 70.55  , 'dt': 0.66499171, 'file': 'd2650um_b0122_b0130.dat'  }, 
                 {'t0': 75.80  , 'dt': 0.76149776, 'file': 'd2650um_b0100_b0122.dat'  }] }

data = spectra[int(sys.argv[1])] if ( (len(sys.argv) > 1) and ( int(sys.argv[1]) in range(10) )) else spectra[0]

slices = len(data)
HD = np.array([np.loadtxt(s['file']) for s in data])
mx = np.reshape( np.repeat(dx, slices) , (slices, dx.size), order='F')

# selected range
fitrange = (dx>= xrmin) & (dx<= xrmax) 
#DX = dx[fitrange]
DX   = mx[: , fitrange]
DSPC = HD[: , fitrange]
#DT =  np.array([ [ d['t0'], d['t0'] + d['dt'] ] for d in data]) 
DT =  np.array([ [ d['t0'], d['t0'] + d['dt'] ] for d in data]) 
DT = DT.T


###############################################################################
## STEP 0
## First fit the curves just to peaks() to retrieve the position and 
## sigma parameters for the background calculation.
##
###############################################################################

def peaks(x, par):
    p = np.reshape(par, (8, x.shape[0]), order='F')
    val =  p[0]/(p[2]*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x.T-p[1])/p[2],2)) + \
           p[3]/(p[5]*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x.T-p[4])/p[5],2)) + \
           p[6] + p[7]*x.T
    return val.T

def fnsqr(x, XX, YY):
    resm = (peaks(XX, x) - YY)/np.sqrt(YY)
    return resm.flatten() 

x0 = slices*guess 

if boundfit:
    bp = bound_pars + [False, False]
    bf = bound_frac + [0.0 , 0.0]
    slcBound = np.array( bp*slices )
    slcFrac  = np.array( bf*slices )
    rc = prevfit[:2]
    prev = np.reshape(prevfit[2:], ( slices , 4)) 
    prevA0 = prevint*(1.0 - ( rc[0] + rc[1]*(DT[0,:] - DT[0,0]) ))
    prevA1 = prevint*( rc[0] + rc[1]*(DT[1,:] - DT[0,0]) )
    
    prevfitbg = np.c_[prevA0, prev[:,0], prev[:,1], prevA1, prev[:,2], prev[:,3], 50*np.ones(slices), 0.01*np.ones(slices)]
    prevfitbg = prevfitbg.flatten()
    x0 = prevfitbg
    bpmin = [ p[1]*(1-p[2]) if p[0] else -np.inf for p in zip(slcBound, prevfitbg, slcFrac) ]
    bpmax = [ p[1]*(1+p[2]) if p[0] else  np.inf for p in zip(slcBound, prevfitbg, slcFrac) ]
    BPARS = (bpmin, bpmax)
    print(f'Bound region for parameters are  {BPARS}')
    res_lsq = least_squares(fnsqr, x0, args=(DX, DSPC), bounds = BPARS)
else:
    res_lsq = least_squares(fnsqr, x0, args=(DX, DSPC))

class fakeres:
    def __init__(self):
        self.x = None
minout = fakeres()
minout.x = res_lsq.x


PAR_step0 = minout.x
print("---- STEP 0 preliminary fit -------\n")
for p in minout.x:
    print(p)

###############################################################################
## STEP 1
## Calculate the background from the histogram 3 sigmas away from both sides,
## employing a loss function for robustify against outliers (contaminats).
## Retrieve the BKG parameters, and the Integral of the peaks.
## A weight function (1 or 0) is employed using the previous fit to mask the 
## region of interest.
## Note: when fiting the bkg outside the peaks, it will be always lower compared
## to the one obtained from a bkg-plus-peaks, as the gaussian peaks wont account 
## for the total Integral due to the degrader, and the fit would rise the bkg 
## when minimizing residuals.
##
###############################################################################
 
def weightbkg(x, par): # just to mask the peaks region (3 sigma)
    p = np.reshape(par, (8, x.shape[0]), order='F')
    lmin = p[1] - 3.0*p[2]
    lmax = p[4] + 3.0*p[5]
    #val =  0.5* ( np.sign((x.T - lmin)*(x.T - lmax)) + 1) 
    val = np.where( (x.T > lmin) & (x.T < lmax), np.zeros(x.T.shape), np.ones(x.T.shape))
    return val.T

def background(x, par):
    p = np.reshape(par, (2, x.shape[0]), order='F')
    val =  p[0] + p[1]*x.T
    return val.T

def getIntegrals(x, y, par, parbkg):
    p  = np.reshape(par, (8, x.shape[0]), order='F')
    bg = np.reshape(parbkg, (2, x.shape[0]), order='F')
    nsig = 3.0
    ## for the range, we should compute the max between p1+3*p2 and p4+3*p5
    lmin = p[1] - nsig*p[2]
    lmax = p[4] + nsig*p[5]
    inside =  0.5* ( 1 - np.sign((x.T - lmin)*(x.T - lmax)) ) 
    below = bg[0] + bg[1]*x.T
    val = inside*(y.T - below)
    valtot = (y.T - below)  ## just take the whole Integral (avoid centroid definition and sigmas)
    print(f'{nsig} region Integrals: {chwidth*val.sum(0)}')
    print(f'Total  region Integrals: {chwidth*valtot.sum(0)}')
    return chwidth*val.sum(0)


def fnmbkg(x, XX, YY, WW, scaling = 1.0):
    #resm = WW*(YY - background(XX, x))
    #resm = WW*(YY - background(XX, x))/np.sqrt(YY)
    resm = WW*(YY - background(XX, x))/scaling
    #rm = resm/resm.size
    #rm = resm/np.sqrt(YY) #scaling factor depends on this normalization
    # rm = resm #scaling factor depends on this normalization
    #r2 = 0.5*rm*rm
    # c = scaling**2.0
    # rsoft = c * ((1 + r2/c)**0.5 - 1 )
    # return 0.5 * rsoft.sum()/WW.sum() 

    # Huber loss robustified...
    loss = np.where( resm < scaling , 0.5*resm*resm, scaling*(np.abs(resm) -0.5*scaling) )

    # now only considering positive outliers
    #loss = np.where( (resm > scaling)&(resm > 0)  , scaling*(np.abs(resm) -0.5*scaling), 0.5*resm*resm )
    return  loss.sum()/WW.sum() 

bkgRMS = 5.0 # estimation of RMS for loss function
par = np.reshape(minout.x, (8, slices), order='F')
guessbkg = par[ [6,7] ,:].flatten(order='F')
bkgW = weightbkg(DX, minout.x)
minout = minimize(fnmbkg, guessbkg , args=(DX, DSPC, bkgW, bkgRMS), method='SLSQP',options = {'maxiter': 500,'disp': True, 'ftol': 1e-10})

for p in minout.x:
    print(p)

PAR_step1 = minout.x

HI = getIntegrals(DX, DSPC, PAR_step0, PAR_step1) ## Histogram Integrals, removed BKG
print(f'Integrals for the slices are {HI}')

###############################################################################
## STEP 2
## Fit the peaks with a 6-parameter function (background fixed), the lateral-bias
## weight, and the residuals for dRQ
## ITERATE at leat 2 times to run once with weighted proper parameters
##
###############################################################################


def peaks4I(x, rr, TT, par, bkg, II):
    r = rr[0] + rr[1]*(TT - TT[0,0]) #+ 0.5*rr[2]*(TT - TT[0,0])*(TT - TT[0,0]) 
    p = np.reshape(par, (4, x.shape[0]), order='F')
    b = np.reshape(bkg, (2, x.shape[0]), order='F')
    Integral = np.reshape(II, (1, x.shape[0]), order='F')
    val =  Integral*(1.0 - r[0])/(p[1]*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x.T-p[0])/p[1],2)) + \
           Integral*r[1]/(p[3]*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x.T-p[2])/p[3],2)) + \
           b[0] + b[1]*x.T
    return val.T


def fnmI(x, XX, YY, TT, bkg, II):
    resh = (YY - peaks4I(XX, x[:2] , TT,  x[2:] , bkg, II)) #/ np.sqrt(YY)

    # Here smoothing the residuals to trim, to reduce noise
    #reshi = np.convolve(reshi, np.ones(8,)/8, mode='same' )
    #np.apply_along_axis(lambda m: np.convolve(m, np.ones(8,)/8, mode='same'), axis=1, arr= reshi)
    p = np.reshape(x[2:], (4, len(x[2:])//4), order='F')
    mzeros = np.zeros(XX.T.shape)
    mones  = np.ones(XX.T.shape)
    mski = np.where( (XX.T >= p[0]+trim_sigmas*p[1]) & (XX.T <= p[2]-trim_sigmas*p[3]), mones, mzeros )
    mski = mski.T
    reshi = resh*mski 

    resh_trim_sum = np.cumsum(reshi, dtype=float, axis=1)
    resh_trim_sum[:, trim:] = resh_trim_sum[:, trim:] - resh_trim_sum[:, :-trim]
    trimresh = resh_trim_sum[:, trim-1:]
    trimresh_sorted = np.sort(trimresh, axis=1)
    trimresh_sortn = np.argsort(trimresh, axis=1)
    trimn = trimresh_sortn[:, -1]

    nrow, ncol = resh.shape
    rowind = np.arange(nrow)[:, np.newaxis]
    colind = np.arange(ncol)
    rolled_colind = (colind + trimn[rowind])%ncol
    rolled_resh = resh[rowind, rolled_colind]
    ctrim_resh = rolled_resh[:, trim:]
    r2trimmed = 0.5*ctrim_resh*ctrim_resh
    return r2trimmed.sum()


#guess = [500.0 , 1204.0 , 5.0, 500.0, 1222, 2.4, 50.0, 0.01 ]
prepare_x0 = np.reshape(PAR_step0, ( len(PAR_step0)//8, 8)) 
prepare2_x0 = prepare_x0[:, [1,2,4,5]]

prepare2_x0[:,0] = prepare2_x0[:,0] 
prepare2_x0[:,1] = 0.95*prepare2_x0[:,1] 
prepare2_x0[:,2] = prepare2_x0[:,2] 
prepare2_x0[:,3] = 0.95*prepare2_x0[:,3] 

r0 = 0.7*prepare_x0[0,3]/(prepare_x0[0,0] + prepare_x0[0,3]) 
r1 = -0.05
#r2 = -0.00015
#r2 = 0.001
#arr_x0 = np.append(np.array([r0,r1,r2]), prepare2_x0.flatten())
arr_x0 = np.append(np.array([r0,r1]), prepare2_x0.flatten())
x0 = arr_x0.tolist()
print(f'Guess parameters {x0}')


pbound = [[xrmin, xrmax], [2.0 ,12.0], [xrmin,xrmax], [0.5, 5]]
#bound = [[0.0, 1.0],[-0.1, 0.0], [-0.001, 0.001]] + slices*pbound
bound = [[0.0, 1.0],[-0.3, 0.0]] + slices*pbound
print(f'BOUNDS ARE {bound}')

if boundfit:
    bp = np.array( slices*bound_pars)
    bp = np.reshape(bp, (slices, 6))
    bf = np.array( slices*bound_frac)
    bf = np.reshape(bf, (slices, 6))
    prevMat = np.reshape(prevfit[2:], ( slices , 4)) 
    r0min = 0.0
    r0max = 1.0
    r1min = -0.1
    r1max = 0.0
    ## So far setting the r-parameters to default (ignoring user bounds)
    prevmin = np.empty(prevMat.shape)
    prevmax = np.empty(prevMat.shape)
    prevmin[:,0] = bp[:,1]*(1-bf[:,1])*prevMat[:,0] + ~bp[:,1]*xrmin
    prevmax[:,0] = bp[:,1]*(1+bf[:,1])*prevMat[:,0] + ~bp[:,1]*xrmax
    prevmin[:,1] = bp[:,2]*(1-bf[:,2])*prevMat[:,1] + ~bp[:,2]*1.0
    prevmax[:,1] = bp[:,2]*(1+bf[:,2])*prevMat[:,1] + ~bp[:,2]*12.0
    prevmin[:,2] = bp[:,4]*(1-bf[:,4])*prevMat[:,2] + ~bp[:,4]*xrmin
    prevmax[:,2] = bp[:,4]*(1+bf[:,4])*prevMat[:,2] + ~bp[:,4]*xrmax
    prevmin[:,3] = bp[:,5]*(1-bf[:,5])*prevMat[:,3] + ~bp[:,5]*0.5
    prevmax[:,3] = bp[:,5]*(1+bf[:,5])*prevMat[:,3] + ~bp[:,5]*5.0
    pminarr = np.array([r0min, r1min])
    pminarr = np.append(pminarr, prevmin.flatten())
    pmaxarr = np.array([r0max, r1max])
    pmaxarr = np.append(pmaxarr, prevmax.flatten())
    prebound = np.array([pminarr, pmaxarr])
    bound = prebound.T
    ## redoing the guess parameters to fit inside the bounds
    #x0 = prevMat.flatten()
    x0 = np.append(prevfit[:2] , prevMat.flatten())
    print('###############################################')
    print(f'Guess X0  {x0}')
    print(f'Bounds  {bound}')
    print('###############################################')

meth = 'trust-constr'
#meth = 'L-BFGS-B'
minout = minimize(fnmI , x0 , args=(DX, DSPC, DT, PAR_step1, HI), method=meth, jac='3-point', bounds = bound, options = {'maxiter': 1800,'disp': True, 'xtol': 1e-8} ) 

### Iterate with new weight,



PAR_step2 = minout.x
for p in minout.x:
    print(p)


## Save parameters for bounding them in another fit, upon request.
if savefit and not boundfit:
    np.savetxt(fbound, PAR_step2)
    np.savetxt(fintegrals, HI)

###################
##   Final results
resT = np.reshape(PAR_step2[2:], ( slices , 4)) 
res = resT.T
resR = PAR_step2[0] + PAR_step2[1]*(DT[1,:] - DT[0,0]) 
resQ = 1.0 - PAR_step2[0] - PAR_step2[1]*(DT[0,:] - DT[0,0]) 

## preliminary errors based on statistical approach for peak areas
errR = resR * np.sqrt(1/(HI*resR) + 1/HI)
errQ = resQ * np.sqrt(1/(HI*resQ) + 1/HI)

print('\n\n ====== Coordinates Decay Curve =======')
for n, r in enumerate(resR):
    print(f"{'{:.03f}'.format(DT[0,n])}   {'{:.05f}'.format(1.0-resQ[n])}   {'{:.05f}'.format(errQ[n])}" )
    print(f"{'{:.03f}'.format(DT[1,n])}   {'{:.05f}'.format(r)}   {'{:.05f}'.format(errR[n])}" )       

### Results for the STEP0
Rini = prepare_x0[:,3]/(prepare_x0[:,0] + prepare_x0[:,3]) 
Tini = 0.5*(DT[1]+DT[0])

print('\n\n ====== Derivative of Decay curve ======')
tcentroid = Tini*HI/HI.sum()
print(f"{'{:.03f}'.format( tcentroid.sum() )}   {'{:.07f}'.format(PAR_step2[1])} ")

print('\n ======= Slice results without Integral =======')
for t,r in zip(Tini, Rini):
    print(f'{t}   {r}')

### slice to plot
s = 1

ffpar = np.append( PAR_step2[:2] ,  PAR_step2[4*s+2:4*(s+1)+2] )
r = ffpar[0] + ffpar[1]*(DT - DT[0,0]) 
fpar = [ (1.0-r[0,s])*HI[s] ,
         ffpar[2] ,
         ffpar[3] ,
         r[1,s]*HI[s] ,
         ffpar[4] ,
         ffpar[5] ]

bpar = PAR_step1[2*s:2*(s+1)]
fpar = fpar +bpar.tolist()

print(f'Integrals for the slices are {HI}')

######################
##   Plotting 

def peaks0(x, p):
    return p[0]/(p[2]*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x-p[1])/p[2],2)) + \
           p[3]/(p[5]*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x-p[4])/p[5],2)) + \
           p[6] + p[7]*x

def background0(x, p):
    return  p[0] + p[1]*x

fig, (spc, decay) = plt.subplots(1,2, gridspec_kw={'width_ratios': [3,2]})
#spc = plt.subplot2grid((1,2), (0,0) ) 
xpl = np.linspace(xrmin, xrmax, 2000)
spc.plot(xpl, peaks0(xpl, fpar), color='tomato')

# plot first attemp from STEP0
spc.plot(xpl, peaks0(xpl, PAR_step0[8*s:8*(s+1)]), color='black')

#spc.plot(xpl, background0(xpl, bpar), color='green')
spc.plot(dx,HD[s,:], drawstyle='steps-mid', color='#121212', linewidth=0.5)

#plot residuals
bkgfull = fpar[6] + fpar[7]*dx
bkgRMS = np.sqrt(bkgfull)
spc.fill_between(x = dx, y1= bkgRMS, y2= -bkgRMS, color='#f0eef0')

reshis = DSPC[s,:] -peaks0(DX[s,:], fpar)
#reshis = np.convolve(20*reshis/np.sqrt(DSPC[s,:]), np.ones(12,)/12, mode='same' )

spc.plot(DX[s,:], reshis , color='blue', linewidth=0.7)

## Show which bins have been trimed..
reshs = DSPC[s,:] -peaks0(DX[s,:], fpar)

## Contiguous trimmed points, new method
mskis = np.where( (DX[s,:] >= fpar[1]+trim_sigmas*fpar[2]) & (DX[s,:] <= fpar[4]-trim_sigmas*fpar[5]), 
                 np.ones(DX[s,:].size), np.zeros(DX[s,:].size) )
reshsw = reshs*mskis
resh_trim_sum = np.cumsum(reshsw, dtype=float)
resh_trim_sum[trim:] = resh_trim_sum[trim:] - resh_trim_sum[:-trim]
trimresh = resh_trim_sum[trim-1:]
trimresh_sorted = np.sort(trimresh)
trimresh_sortn = np.argsort(trimresh)
trimn = trimresh_sortn[-1]
trimpoints = np.arange(trimn, trimn+trim)

xxs = DX[s, trimpoints]
yys = reshs[trimpoints]

#reshsws = np.sort(reshsw)
#reshswsn = np.argsort(reshsw)
##reshit = reshis[: , :-trim]  ## NOTE: we are triming positive residuals only
#xxs = np.take_along_axis(DX[s,:], reshswsn, axis=0)
#trim_points  = xxs[-trim:]
#trim_points_y = np.zeros( xxs[-trim:].size )
#spc.plot(trim_points, trim_points_y , 'bo', markersize=3.5)
spc.plot(xxs, yys , 'bo', markersize=3.5)
##########################

spc.axhline(y=0, linewidth=0.8, linestyle='dashed', color='darkorange')

spc.set_xlim([1150, 1280])
spc.set_ylim(top=1.1*DSPC[s,:].max())
spc.set_xlabel('E (keV)')
spc.set_ylabel('Counts / 0.5 keV')
spc.minorticks_on()

### guide decay approx (quadratic)
def decaycurve(x, a,b,c):
    return a + b*x + c*x*x
decx = np.append(DT[0], DT[1])
decy = np.append(1.0-resQ, resR)
popt, pcov = curve_fit(decaycurve, decx, decy )
xt = np.linspace(DT[0,0], DT[1,slices-1], 1000)
decay.plot(xt, decaycurve(xt, *popt), '-',color='dimgrey', linewidth=0.8 )
################################
#decay = plt.subplot2grid((1,2), (0,1) )

#decay.plot(DT[0], 1.0-resQ, 's', color='tomato', label='1-Q')
#decay.plot(DT[1], resR, 'D', color='royalblue', label='R')
decay.errorbar(DT[0], 1.0-resQ, yerr=errQ, fmt='o', color='tomato', ecolor='grey', elinewidth=0.6, capsize=5, label='1-Q')
decay.errorbar(DT[1], resR, yerr=errR, fmt='o', color='royalblue', ecolor='grey',elinewidth=0.6, capsize=5, label='R')

# original R values from STEP0
decay.plot(Tini, Rini, '--ks', linewidth=0.6 , label='Init')

decay.axvspan(DT[0,s], DT[1,s], color='gainsboro', alpha=0.9)


decay.set_xlabel('Time (ps)')
decay.minorticks_on()
decay.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.92,0.96))

plt.tight_layout()
plt.show()


