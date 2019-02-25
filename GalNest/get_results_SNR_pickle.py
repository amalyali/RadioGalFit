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
import pandas as pd
import pickle

from  montblanc.config import RimeSolverConfig as Options

ARCS2RAD = np.pi/648000.
l, m, s, a, e1, e2 = 0, 1, 2, 3, 4, 5  # define parameter indexes

parser = argparse.ArgumentParser(description='OBSERVED VISIBILITIES SIMULATION')
parser.add_argument('msfile', help='Input MS filename')
args = parser.parse_args(sys.argv[1:])

seed = 1
best_modes_pkl = './data/low_snr/seed%s_14000_0.8_0.1_.pkl' % seed
results_pkl = './data/low_snr/seed%s_14000_0.8_0.1_wSNR.pkl' % seed

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
    # Set visibility noise variance (muJy)
    slvr.set_sigma_sqrd(sigma)

    # Initialise variables for Montblanc slvr.
    nsrc, nssrc, ntime, nbl, nchan = slvr.dim_local_size('nsrc', 'nssrc', 'ntime', 'nbl', 'nchan')

    lm = np.empty([1,2])
    stokes = np.empty(shape=slvr.stokes.shape, dtype=slvr.stokes.dtype)
    I = stokes[:,:,0]
    alpha = slvr.ft(np.ones(nssrc*ntime)*(-0.7)).reshape(nsrc,ntime)
    slvr.transfer_alpha(alpha)

    # Read pickled MultiNest output.
    data = pd.read_pickle(best_modes_pkl)
    modes_SNR = []

    for index, row in data.iterrows():
        lm[:,0] = row['mean'].str[l].values
        lm[:,1] = row['mean'].str[m].values
        slvr.transfer_lm(lm)

        flux = row['mean'].str[s].values
        I[:] = np.outer(flux, np.ones((ntime,)))
        slvr.transfer_stokes(stokes)

        e1 = row['mean'].str[e1].values
        e2 = row['mean'].str[e2].values
        R = row['mean'].str[a].values * ARCS2RAD
        sersic_shape = slvr.ft(np.array([e1,e2,R])).reshape((3,nssrc))
        slvr.transfer_sersic_shape(sersic_shape)

        # Create observed data and upload it to the GPU.
        slvr.solve()
        with slvr.context:
            observed = slvr.retrieve_model_vis()

        # Compute the SNR of the mode.
        modes_SNR.append(np.sqrt(np.sum(np.square(np.absolute(observed))) / sigma))

    # Add on column to dataframe from MultiNest. Write results to file
    modes_SNR = np.asarray(modes_SNR)
    data['SNR'] = modes_SNR
    with open(results_pkl, 'wb') as f:
        pickle.dump(data, f)
