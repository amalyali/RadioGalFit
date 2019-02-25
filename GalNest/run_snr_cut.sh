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
export CUDA_VISIBLE_DEVICES=2  # select a smaller number of GPUs (montblanc uses all GPUs in node by default)
source activate boost_gcc5

MSFILE='2_SKA-1pol.ms'
SNR_THRESH=8.0

python get_results_SNR.py /share/data1/alm/${MSFILE}/ $SNR_THRESH

source deactivate