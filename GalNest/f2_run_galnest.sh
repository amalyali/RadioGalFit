#!/bin/bash
#PBS -l nodes=1:ppn=5
#PBS -q gpu
#PBS -l walltime=100:00:00
#PBS -N galnest_g3_100
#PBS -e ./pbs_info/${PBS_JOBID}e.txt
#PBS -o ./pbs_info/${PBS_JOBID}o.txt

. /etc/profile.d/modules.sh

module load astro/jun2017/multinest
module load gpu/jun2017/montblanc

export PYTHONPATH=/home/mrivi/.local/lib/python2.7/site-packages/montblanc-0.4.0_alpha3_324_ga3b51fd-py2.7.egg
cd /share/splinter/alm/RadioGalFit/GalNest
export CUDA_VISIBLE_DEVICES=3  # select a smaller number of GPUs (montblanc uses all GPUs in node by default)
source activate boost_gcc5

# Define number of galaxies and MultiNest params
NGAL=20
SEED=1
S_EFF=0.8
EV_TOL=0.1
N_LIVE=8000

# Compute observed visibilities
CATALOG='./data/f2_modes/catalog-S10_mal.txt'
MSFILE='/share/data1/alm/10_SKA1-1pol.ms/'
python compute_obs_vis.py ${MSFILE} -ns $NGAL ${CATALOG}

# Standard GalNest run
# Getting example F2 modes
SEED=1
mpiexec -n 1 python GalNest-v4_f2modes.py ${MSFILE} -ns 1 $SEED $N_LIVE $S_EFF $EV_TOL
SEED=2
mpiexec -n 1 python GalNest-v4_f2modes.py ${MSFILE} -ns 1 $SEED $N_LIVE $S_EFF $EV_TOL
SEED=3
mpiexec -n 1 python GalNest-v4_f2modes.py ${MSFILE} -ns 1 $SEED $N_LIVE $S_EFF $EV_TOL
SEED=4
mpiexec -n 1 python GalNest-v4_f2modes.py ${MSFILE} -ns 1 $SEED $N_LIVE $S_EFF $EV_TOL

source deactivate