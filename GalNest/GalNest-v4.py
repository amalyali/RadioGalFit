"""
Detect and estimate parameters of galaxies from observed visibilities contained in a Measurement Set
A. Malyali 2018
"""
from __future__ import absolute_import, unicode_literals
import sys
import argparse
import numpy as np
import scipy.special
from scipy.stats import rv_continuous
import pickle
import pymultinest
from pymultinest.solve import solve
import montblanc
import montblanc.util as mbu

PI = np.pi
C0 = 299792458.0
ARCS2RAD = PI / 648000.
FOV = 3600. * ARCS2RAD

# Parameters to be fitted.
parameters = ["l", "m", "flux", "scalelength", "ee1", "ee2"]

# Define limits of prior.
e_lower, e_upper = 0.0, 0.804
s_lower, s_upper = 3.0, 200.0
scale_lower, scale_upper = 0.3, 3.5  # arcsec

# Load in params from cmd line for output naming.
parser = argparse.ArgumentParser(description='GalNest')
parser.add_argument('msfile', help='Input MS filename')  # MS = measurement set
parser.add_argument('-ns', dest='nssrc', type=int, default=1, help='Number of Sersic Galaxies')
parser.add_argument('seed', help='Seed for MultiNest')
parser.add_argument('n_live', help='No. live points')
parser.add_argument('s_eff', help='Sampling efficiency of multinest')
parser.add_argument('z_tol', help='Evidence tolerance (convergence criterion)')
args = parser.parse_args(sys.argv[1:])

SEED = int(args.seed)
N_LIVE = int(args.n_live)
S_EFF = float(args.s_eff)
EV_TOL = float(args.z_tol)
N_PARAMS = len(parameters)
MAX_MODES= 1000  # for MultiNest to detect
DATA_DIR = './data/low_snr'
PREFIX = '%s/mn_output/galnest_seed%s_%s_%s_%s_' % (DATA_DIR, SEED, N_LIVE, S_EFF, EV_TOL)


# ----------------------------------------------------------------------------------------------------------------------
# Define prior distributions required for MultiNest sampling routine
def flux_CDF(x):
    """
    Cumulative distribution function of a power law flux prior, whose CDF is well know: x^{alpha + 1}/(alpha + 1).
    :param x: value at which CDF is evaluated.
    """
    class flux_pdf(rv_continuous):
        def _pdf(self, y):
            return (-0.34 / (np.power(s_upper, -0.34) - np.power(s_lower, -0.34))) * np.power(y, -1.34)

    flux = flux_pdf(name='flux')
    flux.a, flux.b = s_lower, s_upper  # set range over which CDF defined.
    return flux.cdf(x)


def lognormal_CDF(x):
    """
    CDF of log normal function for scale-lengths.
    :param x: value at which CDF is evaluated.
    """
    mu, sigma = 0.266, 0.3136
    return 0.5 * (1.0 + scipy.special.erf((np.log(x) - mu) / (sigma * np.sqrt(2.0))))


def ellipticity_CDF(x):
    """
    CDF of ellipticity distribution, factors defined in eq 8 of Rivi and Miller 2018.
    :param x: value at which CDF is evaluated.
    """
    e_max = 0.804  # ellipticity cutoff
    e_0 = 0.0732   # circularity parameter
    a = 0.2298  # dispersion
    A = 2.595  # normalization factor

    class ellipticity_pdf(rv_continuous):
        def _pdf(self, y):
            return A * y * (1. - np.exp((y - e_max) / a)) / ((1. + y) * np.sqrt(y * y + e_0 * e_0))

    ellip = ellipticity_pdf(name='ellipticity')
    ellip.a, ellip.b = e_lower, e_upper  # set range over which CDF defined.
    return ellip.cdf(x)


def evaluate_CDF(min_value, max_value, CD_func):
    """
    Evaluate (only done once) CDF between min_value and max_value.
    :param min_value: lower bound of CDF
    :param max_value: upper bound of CDF
    :return: an array of computed CDF between min and max values.
    """
    N = 1000.0

    h = (max_value - min_value) / N  # compute increment size
    CDmin = CD_func(min_value)

    F = CD_func(np.arange(min_value + h, min_value + (N + 1) * h, h)) - CDmin
    F = np.insert(F, 0, 0.0)
    return F


def generate_random_data(u, min_value, max_value, F):
    """
    Generate random data between [min,max] according to a given Cumulative Distribution
    Function (CDF), using inverse transform sampling.
    :param u: random number between 0 and 1
    :param min_value: lower bound of CDF
    :param max_value: upper bound of CDF
    :param F: CDF array, computed using evaluate_CDF.
    :return: a random sample from the probability distribution underlying the inputed CDF.
    """
    i, N = 1, 1000.0
    h = (max_value - min_value) / N  # compute increment size

    while u > F[i] and i <= int(N - 1):
        i += 1

    y = min_value + h * (i - 1) + h * (u - F[i - 1]) / (F[i] - F[i - 1])  # find F^{-1}(u) via linear interpolation
    return y


# ----------------------------------------------------------------------------------------------------------------------
# Begin Montblanc configuration
global slvr, stokes

slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
                                     sources=montblanc.sources(point=0, gaussian=0, sersic=args.nssrc),
                                     init_weights=None, weight_vector=False,
                                     sersic_gradient=False, dtype='double', version='v4')

with montblanc.rime_solver(slvr_cfg) as slvr:  # Read in observed visibilities
    ntime = slvr.dim_local_size('ntime')
    stokes = np.empty(shape=slvr.stokes.shape, dtype=slvr.stokes.dtype)
    I = stokes[:, :, 0]
    alpha = slvr.ft(np.ones(1 * ntime) * (-0.7)).reshape(1, ntime)
    slvr.transfer_alpha(alpha)

    # Let slvr know noise on visibilities, set visibility noise variance (muJy)
    time_acc = 60
    efficiency = 0.9
    channel_bandwidth_hz = 240e6
    SEFD = 400e6
    sigma = (SEFD * SEFD) / (2. * time_acc * channel_bandwidth_hz * efficiency * efficiency)
    slvr.set_sigma_sqrd(sigma)

    # Obtain data visibilities.
    observed_vis = slvr.retrieve_observed_vis()

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate CDF arrays
    ellipticity_CDF_ev = evaluate_CDF(e_lower, e_upper, ellipticity_CDF)
    flux_CDF_ev = evaluate_CDF(s_lower, s_upper, flux_CDF)
    scale_CDF_ev = evaluate_CDF(scale_lower, scale_upper, lognormal_CDF)

    # ------------------------------------------------------------------------------------------------------------------
    # Define prior and likelihood functions required for MultiNest.
    def myprior(q):
        """
        :param q: q = [l, m, flux, scalelength, ee1, ee2]
        :return: list of 6 values drawn randomly from each prior distribution.
        """
        # Position prior
        # Uniformly distributed positions between [-FOV/2 , FOV/2]
        q[0] = (q[0] - 0.5) * FOV
        q[1] = (q[1] - 0.5) * FOV

        # Flux prior
        # Distribution based on VLA observations
        # See Owen and Morrison 2008 for details
        q[2] = generate_random_data(q[2], s_lower, s_upper, flux_CDF_ev)

        # Scalelength prior
        # Distribution based on VLA observations
        q[3] = generate_random_data(q[3], scale_lower, scale_upper, scale_CDF_ev)

        # Ellipticity prior
        # e modulus distribution from SDSS disk-dominated galaxies
        # See Miller et al. 2013 for details
        e_mod = generate_random_data(q[4], e_lower, e_upper, ellipticity_CDF_ev)
        theta = 2 * PI * q[5]
        q[4] = e_mod * np.cos(theta)
        q[5] = e_mod * np.sin(theta)
        return q


    def myloglikelihood(q):
        """
        :param q: q = [l, m, flux, scalelength, ee1, ee2]
        """
        # Position
        lm = mbu.shape_list(q[:2], shape=slvr.lm.shape, dtype=slvr.lm.dtype)
        slvr.transfer_lm(lm)

        # Brightness matrix for source in muJy
        I[:] = np.outer(q[2], np.ones((ntime,)))
        slvr.transfer_stokes(stokes)

        # Shapes
        slvr.transfer_sersic_shape(slvr.ft(np.array([q[4], q[5], q[3] * ARCS2RAD])).reshape((3, 1)))
        slvr.solve()
        return -0.5 * slvr.X2


    # ------------------------------------------------------------------------------------------------------------------
    # Run MultiNest
    result = solve(myloglikelihood,
                   myprior,
                   n_live_points=N_LIVE,
                   seed=SEED,
                   evidence_tolerance=EV_TOL,
                   sampling_efficiency=S_EFF,
                   n_dims=N_PARAMS,
                   outputfiles_basename=PREFIX,
                   resume=False,
                   verbose=False,
                   importance_nested_sampling=False,
                   multimodal=True,
                   n_clustering_params=2,
                   mode_tolerance=-1E+90,
                   max_modes=MAX_MODES)

    # Fetch stats
    a = pymultinest.Analyzer(n_params=N_PARAMS, outputfiles_basename=PREFIX)
    s = a.get_stats()

    mode_stats = a.get_mode_stats()

    # Write results to file
    results_pickle = '%s/seed%s_%s_%s_%s_.pkl' % (DATA_DIR, SEED, N_LIVE, S_EFF, EV_TOL)
    with open(results_pickle, 'wb') as f:
        pickle.dump(mode_stats, f)

    """#with open(file_w, 'a') as txt:
        a = 0
        while a < 2000:
            txt.write("%3.6f %3.6f %3.6f " % (np.asarray(s['modes'][a]['local log-evidence']),
                                              np.asarray(s['modes'][a]['strictly local log-evidence']),
                                              np.asarray(s['modes'][a]['local log-evidence error'])))
            for name, mean, sigma, max_like, max_post in zip(parameters, np.asarray(s['modes'][a]['mean']).transpose(),
                                                             np.asarray(s['modes'][a]['sigma']).transpose(),
                                                             np.asarray(s['modes'][a]['maximum']).transpose(),
                                                             np.asarray(s['modes'][a]['maximum a posterior']).transpose()):
                txt.write("%3.10f %3.10f %3.10f %3.10f " % (mean, sigma, max_like, max_post))

            txt.write("\n")
            a += 1"""
