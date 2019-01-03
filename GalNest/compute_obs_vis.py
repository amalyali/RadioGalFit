#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Marzia Rivi
#

# Input: filename - Measurement Set file
#        ns - number of sersic sources

import sys
import argparse
import pyrap.tables as pt
import numpy as np
import montblanc
import montblanc.util as mbu

ARCS2RAD = np.pi/648000.

#scalelength in arcsec
minscale = 0.3*ARCS2RAD
maxscale = 3.5*ARCS2RAD

parser = argparse.ArgumentParser(description='OBSERVED VISIBILITIES SIMULATION')
parser.add_argument('msfile', help='Input MS filename')
parser.add_argument('-ns',dest='nssrc', type=int, default=1, help='Number of Sersic Galaxies')
parser.add_argument('catfile', help='Input catalogue filename')
args = parser.parse_args(sys.argv[1:])

catalog_file_name = str(args.catfile)

# Get the RIME solver
slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=0, gaussian=0, sersic=args.nssrc),
        init_weights=None, weight_vector=False,
        dtype='double', version='v5')

with montblanc.rime_solver(slvr_cfg) as slvr:
    nsrc, nssrc, ntime, nbl, nchan = slvr.dim_local_size('nsrc', 'nssrc', 'ntime', 'nbl', 'nchan')
    np.random.seed(14352)

    # Read from catalog file, sources flux, scalelength and ellipticity
    catalog = np.loadtxt(catalog_file_name)[:nssrc,:]

    # Random source coordinates in the l,m (brightness image) domain
    l = catalog[:,0]
    m = catalog[:,1]

    lm=mbu.shape_list([l,m], shape=slvr.lm.shape, dtype=slvr.lm.dtype)
    slvr.lm[:,0] = l
    slvr.lm[:,1] = m
    print slvr.lm
    # Brightness matrix for sources
    I = slvr.stokes[:,:,0]
    I[:] = np.outer(catalog[:,2], np.ones((ntime,)))
    slvr.alpha = slvr.ft(np.ones(nssrc*ntime)*(-0.7)).reshape(nsrc,ntime)

    # If there are sersic sources, create their
    # shape matrix and transfer it.
    e1 = catalog[:,4]
    e2 = catalog[:,5]
    R = catalog[:,3]*ARCS2RAD
    slvr.sersic_shape[:] = slvr.ft(np.array([e1,e2,R])).reshape((3,nssrc))
    print np.array([e1,e2,R])
    print slvr.sersic_shape
    #Set channels frequencies
    #slvr.frequency = np.array([1.07e9])

    #Set visibility noise variance (muJy)
    time_acc = 60
    efficiency = 0.9
    channel_bandwidth_hz = 240e6
    SEFD = 400e6
    sigma = (SEFD*SEFD)/(2.*time_acc*channel_bandwidth_hz*efficiency*efficiency)
    slvr.set_sigma_sqrd(sigma)

    # Create observed data and upload it to the GPU
    slvr.solve()
    observed = slvr.model_vis[:].copy()

    print slvr

    noiseR = np.random.normal(0,np.sqrt(sigma),size=observed.shape)
    noiseI = np.random.normal(0,np.sqrt(sigma),size=observed.shape)
    noise = noiseR +1j*noiseI
    observed = observed+noise

    # Open the main table
    main_table = pt.table(args.msfile, readonly=False)

    main_table.putcol('DATA',observed.reshape(ntime*nbl,nchan,1))
    main_table.putcol('CORRECTED_DATA',observed.reshape(ntime*nbl,nchan,1))
    vis = main_table.getcol('CORRECTED_DATA')

    # Close all the tables
    main_table.close()