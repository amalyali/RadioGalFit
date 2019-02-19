#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Marzia Rivi
#

# Input: filename Measurement Set file
#        SNR threshold 
#
# To be run after clustering modes with ml_mode_filtering.py

import sys
import argparse
import math
import pyrap.tables as pt
import numpy as np
import montblanc
import montblanc.util as mbu

from  montblanc.config import RimeSolverConfig as Options

ARCS2RAD = np.pi/648000.

#scalelength in arcsec

parser = argparse.ArgumentParser(description='OBSERVED VISIBILITIES SIMULATION')
parser.add_argument('msfile', help='Input MS filename')
parser.add_argument('threshold', help='SNR threshold')

args = parser.parse_args(sys.argv[1:])
threshold = float(args.threshold)
print "Threshold: ", threshold

#Set visibility noise variance (muJy)
time_acc = 60
efficiency = 0.9
channel_bandwidth_hz = 240e6
SEFD = 400e6
sigma = (SEFD*SEFD)/(2.*time_acc*channel_bandwidth_hz*efficiency*efficiency)

# Get the RIME solver
slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=0, gaussian=0, sersic=1),
        init_weights=None, weight_vector=False,  
        dtype='double', version='v4')

with montblanc.rime_solver(slvr_cfg) as slvr:

    nsrc, nssrc, ntime, nbl, nchan = slvr.dim_local_size('nsrc', 'nssrc', 'ntime', 'nbl', 'nchan')
    
    # Read from modes file, sources flux, scalelength and ellipticity
    data = np.loadtxt('best_modes.txt')
    modes = []    
    modes_SNR = [] 

    lm = np.empty([1,2])
    stokes = np.empty(shape=slvr.stokes.shape, dtype=slvr.stokes.dtype)
    I = stokes[:,:,0]
    alpha = slvr.ft(np.ones(nssrc*ntime)*(-0.7)).reshape(nsrc,ntime)
    slvr.transfer_alpha(alpha)

    #Set visibility noise variance (muJy)
    slvr.set_sigma_sqrd(sigma)

    for i in range(len(data)):
        source = data[i:i+1,:]

        lm[:,0] = source[:,3] 
        lm[:,1] = source[:,7]
        slvr.transfer_lm(lm)

        flux = source[:,11]        
        I[:] = np.outer(flux, np.ones((ntime,)))
        slvr.transfer_stokes(stokes)

        e1 = source[:,19]
        e2 = source[:,23]
        R = source[:,15]*ARCS2RAD
        sersic_shape = slvr.ft(np.array([e1,e2,R])).reshape((3,nssrc))
        slvr.transfer_sersic_shape(sersic_shape)

        # Create observed data and upload it to the GPU
        slvr.solve()
        with slvr.context:
            observed = slvr.retrieve_model_vis()

        SNR = np.sqrt(np.sum(np.square(np.absolute(observed)))/sigma)
        print "flux: ", flux, "scale: ", source[:,15], "SNR: ",SNR

        if SNR > threshold:
            modes.append(i)
            modes_SNR.append(SNR)

        i = i+1

    results = np.empty((len(modes),25))
    results[:,:24] = data[modes,3:]
    results[:,24] = np.array(modes_SNR)
    np.savetxt("results.txt",results)
