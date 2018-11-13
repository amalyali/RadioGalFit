#######################################################################
# Radio Galaxy shape measurement on GPU with HMC  (2018)              #
# Author: Marzia Rivi                                                 #
#                                                                     #
# Input: filename - Measurement Set file                              #
#       ns - number of sersic sources                                 #
#       N  - HMC chain length                                         #
#                                                                     #
# Visibilities are simulated using Montblanc (version 5 - many GPUs)  #
# (https://github.com/ska-sa/montblanc/tree/chi sqrd gradient)        #
#                                                                     #
# HMC is computed using mpp_module (by Sreekumar Thaithara Balain)    #
#                                                                     #
#######################################################################

import sys
import argparse
import math
import mpp_module
import time
import numpy as np
import montblanc
import montblanc.util as mbu

from montblanc.config import RimeSolverConfig as Options

parser = argparse.ArgumentParser(description='HMC-LENSING')
parser.add_argument('msfile', help='Input MS filename')
parser.add_argument('-ns',dest='nssrc', type=int, default=1, help='Number of Sersic Galaxies')
parser.add_argument('-N',dest='N', type=int, default=1, help='MCMC chain length')

args = parser.parse_args(sys.argv[1:])

C0 = 299792458.0
ARCS2RAD = np.pi/648000.

#Defines the prior range
maxlm=np.pi/180.
minlm=-np.pi/180.

#flux in muJy
maxI = 200.
minI = 5.

#scalelength in arcsec
minscale = 0.3*ARCS2RAD
maxscale = 3.5*ARCS2RAD

#scalelength prior
std_v = 0.3136*ARCS2RAD 

#ellipticity prior parameters as in Rivi & Miller (2018)
e_max = 0.804   # ellipticity cutoff
e_0 = 0.0732    # circularity parameter
a = 0.2298      # dispersion
Nfactor = 2.595 # normalization factor


# Get the BIRO solver (from MontBlanc)
# npsrc : number of point sources
# ngsrc : number of gaussian sources
# nssrc : number of sersic sources
# init_weights : (1) None (2) 'sigma' or (3) 'weight'. Either
#   (1) do not initialise the weight vector, or
#   (2) initialise from the MS 'SIGMA' tables, or
#   (3) initialise from the MS 'WEIGHT' tables.
# weight_vector : indicates whether a weight vector should be used to
#   compute the chi squared or a single sigma squared value
# nparams : number of sersic parameters with respect which to 
#   compute the chi_squared gradient
# store_cpu : indicates whether copies of the data passed into the
#   solver transfer_* methods should be stored on the solver object

global slvr

slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=0, gaussian=0, sersic=args.nssrc),
        init_weights=None, weight_vector=False,# mem_budget=1024*1024*1024, 
        sersic_gradient=True, dtype='double', version='v5')

with montblanc.rime_solver(slvr_cfg) as slvr:

    nsrc, nssrc, ntime, nchan = slvr.dim_local_size('nsrc', 'nssrc', 'ntime', 'nchan')
    true_values = np.empty((nssrc,6)) # position, flux, scalelength, ellipticity

    # Read from catalog file, sources position, flux, scalelength and ellipticity
    catalog = np.loadtxt('catalog1000.txt')[:nsrc,:]
    
    # Source coordinates in the l,m (brightness image) domain
    l = catalog[:,0]
    m = catalog[:,1]
    lm = mbu.shape_list([l,m], shape=slvr.lm.shape, dtype=slvr.lm.dtype)
    slvr.lm[:] = lm

    # Brightness matrix for sources (no polarization)
    I, Q, U, V = slvr.stokes[:,:,0], slvr.stokes[:,:,1], slvr.stokes[:,:,2], slvr.stokes[:,:,3]
    I[:] = np.outer(catalog[:,2], np.ones((ntime,)))
    Q[:] = np.zeros(shape=Q.shape)
    U[:] = np.zeros(shape=U.shape)
    V[:] = np.zeros(shape=V.shape)
    slvr.alpha[:] = slvr.ft(np.ones(nssrc*ntime)*(-0.7)).reshape(nsrc,ntime)
    
    # mean lognormal prior for scalelength, dependent on source flux (as in Rivi et al 2016)
    mean_v = (-0.93+0.33*np.log(catalog[:,2]))+np.log(ARCS2RAD)

    # If there are sersic sources, create their
    # shape matrix and transfer it.
    if nssrc > 0:
        mod = slvr.ft(np.random.random(nssrc))*0.4
        angle = slvr.ft(np.random.random(nssrc))*2*np.pi
        e1 = catalog[:,4]
        e2 = catalog[:,5]
        R = catalog[:,3]*ARCS2RAD
        sersic_shape_or = slvr.ft(np.array([e1,e2,R])).reshape((3,nssrc))
        slvr.sersic_shape[:] = sersic_shape_or

    # E_beam = mbu.random_like(slvr.E_beam_gpu)
    # slvr.transfer_E_beam(E_beam)
    
    # Set visibility noise variance (muJy)
    time_acc = 60
    channel_bandwidth_hz = 240e6
    efficiency = 0.9
    SEFD_SKA = 400e6
    sigma = (SEFD_SKA*SEFD_SKA)/(2.*time_acc*channel_bandwidth_hz*efficiency*efficiency)
    slvr.set_sigma_sqrd(sigma)
    
    # Create observed visibilities (if not already available in the MS file) and upload them to the GPU
    slvr.solve()
    observed = slvr.model_vis[:].copy()

    np.random.seed(4263817185)
    noiseR = np.random.normal(0,np.sqrt(sigma),size=observed.shape)
    noiseI = np.random.normal(0,np.sqrt(sigma),size=observed.shape)
    noise = noiseR +1j*noiseI
    slvr.observed_vis[:] = (observed+noise).copy()
    print 'transfer to GPU'

    # Generate and transfer a weight vector.
    # weight_vector = mbu.random_like(slvr.weight_vector_gpu))
    # slvr.transfer_weight_vector(weight_vector)

    slvr.solve()

    print slvr


    # Prior
    def logprior(q):
        # ellipticity prior as in Miller et al (2013)
        ee1 = q[:nssrc]
        ee2 = q[nssrc:2*nssrc]
        emod = np.sqrt(ee1*ee1+ee2*ee2)
        if np.any([eval>e_max for eval in emod]):
            return -1e90
        else:
            logp_e = np.log(Nfactor*(emod*(1.-np.exp((emod-e_max)/a))))-np.log(1.+emod)-0.5*np.log(emod*emod+e_0*e_0)

        # scalelength prior
        scale = q[2*nssrc:]
        if np.any([value<=0 for value in scale]):
            return  -1e90
        else:
            x = (np.log(scale)-mean_v)/std_v
            logp_r = -x*x*0.5 -np.log(scale*std_v)
        return np.sum(logp_e)+np.sum(logp_e)


    # Prior Derivatives
    def logpriorDeriv(q):
        # ellipticity prior derivative
        ee1 = q[:nssrc]
        ee2 = q[nssrc:2*nssrc]
        emod = np.sqrt(ee1*ee1+ee2*ee2)
        if np.any([eval>e_max for eval in emod]):
            return 0.
        else:
            x = np.exp((emod-e_max)/a)
            dlogp_e = 1./emod-x/(a*(1.-x))-1./(1.+emod)-emod/(emod*emod+e_0*e_0)

        # scalelength prior derivative
        scale = q[2*nssrc:]
        if np.any([value<=0 for value in scale]):
            return 0.
        else:
            x = (np.log(scale)-mean_v)/std_v
            dlogp_r = (-1./scale)*(x/std_v + 1.)
        return np.sum(dlogp_e)+np.sum(dlogp_r)


    # Posterior
    def logPostFunc(q):
        # Set model on the shared data object. Uploads to GPU
        slvr.sersic_shape[:] = q.reshape((3,nssrc))
        slvr.solve() 
        loglike = -0.5*slvr.X2
        val = loglike+logprior(q)
        return val

    # Posterior Gradient
    def logPostDerivs(q):
        # Set model on the shared data object. Uploads to GPU
        slvr.sersic_shape[:] = q.reshape((3,nssrc))
        slvr.solve()
        dq = -0.5*slvr.X2_grad[:].reshape(3*nssrc) 
        return dq 

    # define the dimensionality of the Gaussian posterior
    numParams = 3*nssrc 

    # Starting points for the sampling
    # assume flux and positions known and take into account Montblanc bug 
    start_e1 = slvr.ft(np.random.random(nssrc))*0.2 - 0.1  
    start_e2 = slvr.ft(np.random.random(nssrc))*0.2 - 0.1 
    start_sersic_scale = np.exp(-0.93+0.33*np.log(catalog[:,2]))*ARCS2RAD   ##linear relation between scale_std[arcsec] and flux[muJy] (Rivi et al 2016)
    startPoint = np.array([start_e1,start_e2,start_sersic_scale], dtype=slvr.ft).reshape(3*nssrc)

    # maximum value of the parameter epsilon; 0<epsilon<2
    maxEps = 0.05

    # maximum number of leapfrog / euler steps
    maxNumSteps = 10

    # seed for the random number generator
    randSeed = 1234

    # number samples to be taken in one interation.
    # how often you would like the chains to be written?
    # every time packet_size samples
    packetSize = 100

    # number samples to be burned
    numBurn = 5000

    # number samples to be taken (after burning)
    numSamples = args.N

    # path to chains and other output
    rootPathStr = "/share/apps/mrivi/HMC_lensing/testHMC"

    # do we require output to console? 0 means NO, !=0 means YES
    consoleOutput = 100

    # delimiter for the chain data
    delimiter = " "

    #  percision with which the chains should be written
    precision = 10

    # inverse of the diagonal kinetic energy mass matrix
    # this should be close/equal to the covariance
    # matrix of parameters / posterior distriubtion if Gaussian
    # If parameters are not correlated, try diagonal elemens

    #linear relation between e_std and flux[muJy] extrapolated from previous tests (Rivi et al 2018)
    steps_e = 0.1525 - 0.00175*catalog[:,2]
    steps_scale = (0.2175 - 0.00225*catalog[:,2])*ARCS2RAD
    steps = slvr.ft(np.concatenate((steps_e,steps_e,steps_scale),axis=0))
    keDiagMInv = steps*steps

    print 'Starting HMC\n'
    T1=time.time()

    mpp_module.canonicalHamiltonianSampler(
        numParams,
        maxEps,
        maxNumSteps,
        startPoint,
        randSeed,
        packetSize,
        numBurn,
        numSamples,
        rootPathStr,
        consoleOutput,
        delimiter,
        precision,
        keDiagMInv,
        logPostFunc,
        logPostDerivs
    )
    print 'Time taken',(time.time()-T1)/60.,'min'


