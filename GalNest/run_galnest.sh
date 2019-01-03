#!/bin/bash
#PBS -l nodes=1:ppn=5
#PBS -q gpu
#PBS -l walltime=100:00:00
#PBS -N galnest_g3_100

. /etc/profile.d/modules.sh

module load astro/jun2017/multinest
module load gpu/jun2017/montblanc

export PYTHONPATH=/home/mrivi/.local/lib/python2.7/site-packages/montblanc-0.4.0_alpha3_324_ga3b51fd-py2.7.egg
cd /share/splinter/alm/shape_measure
export CUDA_VISIBLE_DEVICES=3  # (in order to select a smaller number of GPUs, by default montblanc uses all GPUs in the node)
source activate boost_gcc5

# MultiNest params
NGAL=2
SEED=1
S_EFF=0.8
EV_TOL=0.1
N_LIVE=5000

## Compute observed visibilities
python compute_obs_vis.py /share/data1/alm/10_SKA1-1pol.ms/ -ns $NGAL > ./data/output_comp_vis_single-v4

## Standard GalNest run
mpiexec -n 1 python galnest-v4.py /share/data1/alm/10_SKA1-1pol.ms/ -ns 1 $NGAL $SEED $N_LIVE $S_EFF $EV_TOL > galnest_


source deactivate